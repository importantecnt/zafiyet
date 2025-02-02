from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import re
import zipfile
import tempfile
import shutil
from vulnerability_data import vulnerabilities_extended
import json
import traceback

app = Flask(__name__)

# Model ve tokenizer'ı global olarak yükle
try:
    model_path = './vulnerability_model'
    if os.path.exists(model_path):
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model.eval()
    else:
        print("UYARI: Model dosyası bulunamadı!")
        model = None
        tokenizer = None
except Exception as e:
    print(f"Model yükleme hatası: {str(e)}")
    model = None
    tokenizer = None

def analyze_file_content(file_path, file_content):
    """Dosya içeriğini analiz et ve zafiyetleri tespit et"""
    try:
        vulnerabilities = []
        risk_score = 0
        
        # Temel güvenlik kontrolleri
        security_checks = {
            "SQL Injection": {
                "pattern": r"(?i)(select|insert|update|delete).*?(where|from|into|values)",
                "risk_level": "Kritik",
                "description": "SQL Injection zafiyeti tespit edildi. Bu zafiyet, veritabanı sorgularında kullanıcı girdisinin doğrudan kullanılmasından kaynaklanır."
            },
            "XSS": {
                "pattern": r"<script.*?>|javascript:|innerHTML|document\.write",
                "risk_level": "Yüksek",
                "description": "Cross-Site Scripting (XSS) zafiyeti tespit edildi. Bu zafiyet, kullanıcı girdisinin doğrudan HTML içeriğine eklenmesinden kaynaklanır."
            },
            "File Inclusion": {
                "pattern": r"include\s*\(|require\s*\(|readFile\s*\(|fs\.read",
                "risk_level": "Kritik",
                "description": "Güvensiz dosya işleme zafiyeti tespit edildi. Bu zafiyet, dosya yollarının doğrulanmadan kullanılmasından kaynaklanır."
            },
            "Command Injection": {
                "pattern": r"exec\s*\(|system\s*\(|shell_exec|eval\s*\(",
                "risk_level": "Kritik",
                "description": "Command Injection zafiyeti tespit edildi. Bu zafiyet, sistem komutlarının güvensiz bir şekilde çalıştırılmasından kaynaklanır."
            },
            "Weak Crypto": {
                "pattern": r"md5\s*\(|sha1\s*\(|crypto\.createHash\s*\(\s*['\"]md5['\"]",
                "risk_level": "Yüksek",
                "description": "Zayıf kriptografi kullanımı tespit edildi. MD5 veya SHA1 gibi güvenli olmayan hash algoritmaları kullanılıyor."
            },
            "Hardcoded Credentials": {
                "pattern": r"password\s*=\s*['\"][^'\"]+['\"]|api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
                "risk_level": "Yüksek",
                "description": "Kodda sabit tanımlanmış kimlik bilgileri tespit edildi. Bu durum güvenlik açığına neden olabilir."
            },
            "Insecure Direct Object References (IDOR)": {
                "pattern": r"(?i)\/user\/\d+",
                "risk_level": "Yüksek",
                "description": "Kullanıcıların, yetkileri dışında nesnelere erişim sağlamasına olanak tanıyan bir güvenlik açığıdır."
            },
            "Security Misconfiguration": {
                "pattern": r"(?i)config\s*=\s*['\"]?[^'\";]+['\"]?",
                "risk_level": "Yüksek",
                "description": "Uygulama veya sunucu yapılandırmalarının yanlış yapılması sonucu ortaya çıkan bir güvenlik açığıdır."
            },
            "Sensitive Data Exposure": {
                "pattern": r"(?i)(password|ssn|credit\s*card|api\s*key|secret)\s*=\s*['\"][^'\"]+['\"]",
                "risk_level": "Kritik",
                "description": "Hassas verilerin yeterince korunmaması sonucu ortaya çıkan bir güvenlik açığıdır."
            },
            "Insufficient Logging and Monitoring": {
                "pattern": r"(?i)log\s*=\s*False|logging\s*=\s*False",
                "risk_level": "Yüksek",
                "description": "Uygulama veya sistemde yeterli günlük kaydı ve izleme yapılmaması sonucu ortaya çıkan bir güvenlik açığıdır."
            },
            "Using Components with Known Vulnerabilities": {
                "pattern": r"(?i)import\s+(vulnerable_library|old_library)",
                "risk_level": "Yüksek",
                "description": "Bilinen zafiyetlere sahip bileşenlerin kullanılması."
            },
            "Insufficient Security Controls": {
                "pattern": r"(?i)security\s*=\s*False|disable_security",
                "risk_level": "Yüksek",
                "description": "Yetersiz güvenlik kontrolleri nedeniyle ortaya çıkan zafiyetler."
            }
        }
        
        # Dosya içeriğini satırlara böl
        lines = file_content.split('\n')
        
        # Her güvenlik kontrolü için
        for vuln_name, vuln_info in security_checks.items():
            matches = re.finditer(vuln_info["pattern"], file_content)
            for match in matches:
                # Eşleşen kodun bulunduğu satırı bul
                line_no = file_content[:match.start()].count('\n')
                
                # Eşleşen kodun bulunduğu bölgeyi al (önceki ve sonraki 2 satır)
                start_line = max(0, line_no - 2)
                end_line = min(len(lines), line_no + 3)
                
                # Problemli kod bloğunu oluştur
                vulnerable_code = '\n'.join(lines[start_line:end_line])
                
                # Düzeltme önerisi oluştur
                fix_suggestion = get_fix_suggestion(vuln_name, vulnerable_code)
                
                vulnerabilities.append({
                    "name": vuln_name,
                    "description": vuln_info["description"],
                    "risk_level": vuln_info["risk_level"],
                    "line_number": line_no + 1,
                    "code_sample": vulnerable_code,
                    "fix_suggestion": fix_suggestion,
                    "file_path": file_path,
                    "impacts": get_vulnerability_impacts(vuln_name),
                    "solutions": get_vulnerability_solutions(vuln_name)
                })
                
                # Risk skorunu güncelle
                if vuln_info["risk_level"] == "Kritik":
                    risk_score += 10
                elif vuln_info["risk_level"] == "Yüksek":
                    risk_score += 7
                else:
                    risk_score += 4
        
        # Ortalama risk skorunu hesapla
        risk_score = min(10, risk_score / max(len(vulnerabilities), 1))
        
        return {
            "name": os.path.basename(file_path),
            "vulnerabilities": vulnerabilities,
            "risk_score": round(risk_score, 2),
            "file_content": file_content,
            "file_path": file_path
        }
    except Exception as e:
        print(f"Dosya analiz hatası: {str(e)}")
        traceback.print_exc()
        return {
            "name": os.path.basename(file_path),
            "vulnerabilities": [],
            "risk_score": 0,
            "error": str(e)
        }

def get_fix_suggestion(vuln_type, vulnerable_code):
    """Zafiyet türüne göre düzeltme önerisi oluştur"""
    if vuln_type == "SQL Injection":
        return vulnerable_code.replace(
            "SELECT * FROM", 
            "cursor.execute('SELECT * FROM users WHERE username = %s', (username,))"
        )
    elif vuln_type == "XSS":
        return vulnerable_code.replace(
            "innerHTML", 
            "textContent"
        )
    elif vuln_type == "Command Injection":
        return """# Güvenli alternatif:
import subprocess
subprocess.run(['ls', directory], check=True, shell=False)"""
    elif vuln_type == "Weak Crypto":
        return """# Güvenli alternatif:
import bcrypt
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())"""
    
    return "// Düzeltme önerisi bulunamadı"

def get_vulnerability_impacts(vuln_type):
    """Zafiyet türüne göre olası etkileri döndür"""
    impacts = {
        "SQL Injection": [
            "Veritabanı manipülasyonu",
            "Veri sızıntısı",
            "Yetkisiz erişim",
            "Veri kaybı"
        ],
        "XSS": [
            "Kullanıcı oturumu çalınması",
            "Zararlı kod çalıştırma",
            "Kullanıcı aldatmaca",
            "Veri çalınması"
        ]
    }
    return impacts.get(vuln_type, ["Etki bilgisi bulunamadı"])

def get_vulnerability_solutions(vuln_type):
    """Zafiyet türüne göre çözüm önerilerini döndür"""
    solutions = {
        "SQL Injection": [
            "Parametreli sorgular kullanın",
            "ORM kullanın",
            "Giriş verilerini doğrulayın",
            "En az yetki prensibini uygulayın"
        ],
        "XSS": [
            "HTML karakterlerini escape edin",
            "Content Security Policy (CSP) kullanın",
            "Giriş verilerini temizleyin",
            "XSS koruma kütüphaneleri kullanın"
        ]
    }
    return solutions.get(vuln_type, ["Çözüm önerisi bulunamadı"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'project' not in request.files:
            return jsonify({'error': 'Dosya yüklenmedi'}), 400
        
        project_file = request.files['project']
        if project_file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        if not project_file.filename.endswith('.zip'):
            return jsonify({'error': 'Sadece ZIP dosyaları kabul edilir'}), 400
        
        temp_dir = tempfile.mkdtemp()
        try:
            zip_path = os.path.join(temp_dir, 'project.zip')
            project_file.save(zip_path)
            
            extract_dir = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_dir)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            code_extensions = {'.py', '.js', '.php', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.ts'}
            
            analyzed_files = []
            total_risk_score = 0
            
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in code_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            result = analyze_file_content(file_path, content)
                            if result.get('error'):
                                print(f"Dosya analiz hatası ({file}): {result['error']}")
                            else:
                                analyzed_files.append(result)
                                total_risk_score += result['risk_score']
                        except Exception as e:
                            print(f"Dosya okuma hatası ({file}): {str(e)}")
                            traceback.print_exc()
            
            if not analyzed_files:
                return jsonify({'error': 'Analiz edilebilir dosya bulunamadı'}), 400
            
            project_risk_score = round(total_risk_score / len(analyzed_files), 2)
            
            return jsonify({
                'project_risk_score': project_risk_score,
                'total_vulnerabilities': sum(len(f['vulnerabilities']) for f in analyzed_files),
                'files': analyzed_files
            })
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print("Genel hata:", str(e))
        traceback.print_exc()
        return jsonify({'error': f'Analiz sırasında bir hata oluştu: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
