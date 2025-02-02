import pandas as pd
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# Static klasörünü oluştur
if not os.path.exists('static'):
    os.makedirs('static')

# OWASP Top 10 2021 genişletilmiş zafiyet verileri
vulnerabilities_extended = {
    "A01:2021-Broken Access Control": {
        "name": "Broken Access Control",
        "description": "Erişim kontrolü zafiyeti tespit edildi.",
        "risk_level": "Kritik",
        "impact": [
            "Yetkisiz veri erişimi",
            "Sistem manipülasyonu",
            "Veri sızıntısı",
            "Kullanıcı hesaplarının ele geçirilmesi"
        ],
        "solutions": [
            "Role-based access control (RBAC) implementasyonu",
            "Principle of least privilege uygulanması",
            "Session yönetiminin güçlendirilmesi",
            "JWT token güvenliğinin sağlanması",
            "API güvenlik kontrollerinin implementasyonu"
        ],
        "detection_patterns": [
            r"auth.*bypass",
            r"admin.*direct",
            r"role.*check.*missing",
            r"permission.*validation"
        ]
    },
    "A02:2021-Cryptographic Failures": {
        "name": "Cryptographic Failures",
        "description": "Kriptografik güvenlik zafiyeti tespit edildi.",
        "risk_level": "Kritik",
        "impact": [
            "Veri sızıntısı",
            "Kimlik hırsızlığı",
            "Finansal kayıp",
            "Hassas veri ifşası"
        ],
        "solutions": [
            "Güçlü şifreleme algoritmalarının kullanılması",
            "Güvenli key yönetimi",
            "TLS 1.3 implementasyonu",
            "Güvenli hash fonksiyonlarının kullanımı",
            "Şifreleme anahtarlarının düzenli rotasyonu"
        ],
        "detection_patterns": [
            r"md5\(",
            r"sha1\(",
            r"weak.*encryption",
            r"plain.*text.*password"
        ]
    },
    "A03:2021-Injection": {
        "name": "Injection",
        "description": "Kod enjeksiyon zafiyeti tespit edildi.",
        "risk_level": "Kritik",
        "impact": [
            "Veri tabanı manipülasyonu",
            "Sistem komutlarının çalıştırılması",
            "Kullanıcı verilerinin çalınması",
            "Sistemin ele geçirilmesi"
        ],
        "solutions": [
            "Prepared statements kullanımı",
            "Input validasyonu",
            "Escape karakterlerinin kullanımı",
            "ORM kullanımı",
            "WAF implementasyonu"
        ],
        "detection_patterns": [
            r"exec\(",
            r"eval\(",
            r"system\(",
            r"SELECT.*WHERE.*\$",
            r"INSERT.*VALUES.*\$"
        ]
    },
    "A04:2021-Insecure Direct Object References (IDOR)": {
        "name": "Insecure Direct Object References (IDOR)",
        "description": "Kullanıcıların, yetkileri dışında nesnelere erişim sağlamasına olanak tanıyan bir güvenlik açığıdır.",
        "risk_level": "Yüksek",
        "impact": [
            "Yetkisiz veri erişimi",
            "Veri sızıntısı"
        ],
        "solutions": [
            "Erişim kontrolleri uygulayın",
            "Kullanıcı yetkilerini doğrulayın"
        ],
        "detection_patterns": [
            r"direct.*object.*reference",
            r"unauthorized.*access",
            r"missing.*authorization"
        ]
    },
    "A05:2021-Security Misconfiguration": {
        "name": "Security Misconfiguration",
        "description": "Uygulama veya sunucu yapılandırmalarının yanlış yapılması sonucu ortaya çıkan bir güvenlik açığıdır.",
        "risk_level": "Yüksek",
        "impact": [
            "Hizmetin kötüye kullanılması",
            "Veri sızıntısı"
        ],
        "solutions": [
            "Güvenli yapılandırma standartları uygulayın",
            "Düzenli güvenlik denetimleri yapın"
        ],
        "detection_patterns": [
            r"misconfiguration",
            r"default.*credentials",
            r"unnecessary.*services"
        ]
    },
    "A06:2021-Sensitive Data Exposure": {
        "name": "Sensitive Data Exposure",
        "description": "Hassas verilerin yeterince korunmaması sonucu ortaya çıkan bir güvenlik açığıdır.",
        "risk_level": "Kritik",
        "impact": [
            "Kişisel bilgilerin ifşası",
            "Kimlik hırsızlığı"
        ],
        "solutions": [
            "Verileri şifreleyin",
            "Güvenli veri saklama yöntemleri kullanın"
        ],
        "detection_patterns": [
            r"unprotected.*data",
            r"plaintext.*sensitive.*information",
            r"missing.*encryption"
        ]
    },
    "A07:2021-Missing Function Level Access Control": {
        "name": "Missing Function Level Access Control",
        "description": "Fonksiyon seviyesinde erişim kontrolü eksikliği.",
        "risk_level": "Yüksek",
        "impact": [
            "Yetkisiz erişim",
            "Veri manipülasyonu"
        ],
        "solutions": [
            "Her fonksiyon için erişim kontrolü uygulayın",
            "Kullanıcı rolleri ve izinlerini yönetin"
        ],
        "detection_patterns": [
            r"missing.*function.*level.*access.*control",
            r"unauthorized.*function.*access"
        ]
    },
    "A08:2021-Insufficient Logging and Monitoring": {
        "name": "Insufficient Logging and Monitoring",
        "description": "Uygulama veya sistemde yeterli günlük kaydı ve izleme yapılmaması sonucu ortaya çıkan bir güvenlik açığıdır.",
        "risk_level": "Yüksek",
        "impact": [
            "Saldırıların tespit edilememesi",
            "Zamanında müdahale edilememesi"
        ],
        "solutions": [
            "Günlük kaydı ve izleme sistemleri kurun",
            "Olay müdahale planları oluşturun"
        ],
        "detection_patterns": [
            r"insufficient.*logging",
            r"missing.*monitoring"
        ]
    },
    "A09:2021-Using Components with Known Vulnerabilities": {
        "name": "Using Components with Known Vulnerabilities",
        "description": "Bilinen zafiyetlere sahip bileşenlerin kullanılması.",
        "risk_level": "Yüksek",
        "impact": [
            "Sistem güvenliğinin ihlali",
            "Veri sızıntısı"
        ],
        "solutions": [
            "Bileşenlerin güncel sürümlerini kullanın",
            "Güvenlik güncellemelerini takip edin"
        ],
        "detection_patterns": [
            r"known.*vulnerabilities",
            r"outdated.*components"
        ]
    },
    "A10:2021-Insufficient Security Controls": {
        "name": "Insufficient Security Controls",
        "description": "Yetersiz güvenlik kontrolleri nedeniyle ortaya çıkan zafiyetler.",
        "risk_level": "Yüksek",
        "impact": [
            "Sistem güvenliğinin ihlali",
            "Veri kaybı"
        ],
        "solutions": [
            "Güvenlik kontrollerini güçlendirin",
            "Düzenli güvenlik testleri yapın"
        ],
        "detection_patterns": [
            r"insufficient.*security.*controls",
            r"weak.*security.*measures"
        ]
    }
}

# Genişletilmiş eğitim örnekleri
training_examples = [
    {
        "text": "Kullanıcı kimlik doğrulaması atlatıldı ve admin paneline erişildi",
        "vulnerability": "A01:2021-Broken Access Control",
        "code_sample": """
        if user.is_authenticated:
            return admin_panel()  # Missing role check
        """
    },
    {
        "text": "Şifreler plain text olarak veritabanında saklanıyor",
        "vulnerability": "A02:2021-Cryptographic Failures",
        "code_sample": """
        user.password = request.form['password']  # No encryption
        db.save(user)
        """
    },
    {
        "text": "Login formunda SQL injection açığı tespit edildi",
        "vulnerability": "A03:2021-Injection",
        "code_sample": """
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        """
    },
    {
        "text": "Yetkisiz nesne referansı ile veri erişimi",
        "vulnerability": "A04:2021-Insecure Direct Object References (IDOR)",
        "code_sample": """
        @app.route('/user/<id>')
        def get_user(id):
            return User.query.get(id)  # Missing ownership check
        """
    },
    {
        "text": "Güvenli olmayan yapılandırma kullanımı",
        "vulnerability": "A05:2021-Security Misconfiguration",
        "code_sample": """
        @app.route('/api/data')
        def get_data():
            return jsonify(sensitive_data)  # No authentication check
        """
    },
    {
        "text": "Hassas verilerin şifrelenmeden saklanması",
        "vulnerability": "A06:2021-Sensitive Data Exposure",
        "code_sample": """
        def save_user_data(username, password):
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                      [username, password])  # Plain text password
        """
    },
    {
        "text": "Fonksiyon seviyesinde erişim kontrolü eksik",
        "vulnerability": "A07:2021-Missing Function Level Access Control",
        "code_sample": """
        @app.route('/admin')
        def admin_panel():
            return render_template('admin.html')  # No access control
        """
    },
    {
        "text": "Yetersiz günlük kaydı",
        "vulnerability": "A08:2021-Insufficient Logging and Monitoring",
        "code_sample": """
        def process_request(request):
            log_request(request)  # No logging implemented
        """
    },
    {
        "text": "Bilinen zafiyetlere sahip kütüphane kullanımı",
        "vulnerability": "A09:2021-Using Components with Known Vulnerabilities",
        "code_sample": """
        import vulnerable_library  # Using a library with known vulnerabilities
        """
    },
    {
        "text": "Yetersiz güvenlik kontrolleri",
        "vulnerability": "A10:2021-Insufficient Security Controls",
        "code_sample": """
        def process_payment(payment_info):
            return process(payment_info)  # No security checks
        """
    }
]

class VulnerabilityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, 
                                 truncation=True, 
                                 padding=True, 
                                 max_length=max_length, 
                                 return_tensors='pt')
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class VulnerabilityTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=2e-5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
        return val_loss / len(self.val_loader), predictions, true_labels

def save_vulnerability_data():
    """Zafiyet verilerini JSON olarak kaydet"""
    data = {
        "vulnerabilities": vulnerabilities_extended,
        "training_examples": training_examples
    }
    
    with open('vulnerability_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_vulnerability_data():
    """Zafiyet verilerini JSON'dan yükle"""
    if os.path.exists('vulnerability_data.json'):
        with open('vulnerability_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    return None

def train_and_evaluate(trainer, epochs=3):
    for epoch in range(epochs):
        train_loss = trainer.train_epoch()
        val_loss, predictions, true_labels = trainer.evaluate()
        
        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        trainer.accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nSınıflandırma Raporu:")
        print(classification_report(true_labels, predictions))
        print("-" * 50)
    
    plot_training_results(trainer, true_labels, predictions)

def plot_training_results(trainer, true_labels, predictions):
    try:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(trainer.train_losses, label='Training Loss')
        plt.plot(trainer.val_losses, label='Validation Loss')
        plt.title('Eğitim ve Doğrulama Loss Değerleri')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(trainer.accuracies, label='Accuracy', color='green')
        plt.title('Model Doğruluk Oranı')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.ylabel('Gerçek Sınıf')
        
        plt.tight_layout()
        
        # Dosya yolunu düzelt ve klasörün varlığını kontrol et
        save_path = os.path.join('static', 'training_results.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Grafik kaydedildi: {save_path}")
    except Exception as e:
        print(f"Grafik kaydedilirken hata oluştu: {str(e)}")

if __name__ == "__main__":
    # Klasör yapısını oluştur
    os.makedirs('static', exist_ok=True)
    os.makedirs('vulnerability_model', exist_ok=True)
    
    # Zafiyet verilerini kaydet
    save_vulnerability_data()
    
    # Model ve tokenizer yükleme
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=len(vulnerabilities_extended)
    )

    # Veri hazırlama
    texts = [ex["text"] + "\n" + ex.get("code_sample", "") for ex in training_examples]
    labels = [list(vulnerabilities_extended.keys()).index(ex["vulnerability"]) 
             for ex in training_examples]

    # Veri bölme
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Dataset ve DataLoader oluşturma
    train_dataset = VulnerabilityDataset(train_texts, train_labels, tokenizer)
    val_dataset = VulnerabilityDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Trainer oluşturma ve eğitim
    trainer = VulnerabilityTrainer(model, train_loader, val_loader)
    train_and_evaluate(trainer, epochs=3)

    # Model kaydetme
    model.save_pretrained('./vulnerability_model')
    tokenizer.save_pretrained('./vulnerability_model')

    print("\nModel eğitimi tamamlandı ve kaydedildi.")
