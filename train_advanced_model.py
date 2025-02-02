# Gerekli kütüphanelerin kontrolü ve yüklenmesi
try:
    import torch
    import transformers
    import nltk
    import spacy
except ImportError as e:
    print(f"Eksik kütüphane: {e}")
    print("Lütfen gerekli kütüphaneleri yükleyin:")
    print("pip install torch transformers nltk spacy")
    exit(1)

# NLTK verilerinin kontrolü ve yüklenmesi
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK verileri yükleniyor...")
    nltk.download('punkt')
    nltk.download('stopwords')

# spaCy modelinin kontrolü ve yüklenmesi
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("spaCy modeli yükleniyor...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Gerekli importlar
from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)
from torch.utils.data import (
    Dataset,
    DataLoader
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Local importlar
try:
    from vulnerability_data import (
        vulnerabilities_extended,
        training_examples
    )
except ImportError:
    print("vulnerability_data.py dosyası bulunamadı!")
    exit(1)

# NLTK bileşenleri
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

print("Tüm kütüphaneler başarıyla yüklendi!")


training_examples = [
    # A01:2021-Broken Access Control örnekleri
    {
        "text": "Kullanıcı kimlik doğrulaması atlatıldı ve admin paneline erişildi.",
        "vulnerability": "A01:2021-Broken Access Control",
        "code_sample": """
        if user.is_authenticated:
            return admin_panel()  # Missing role check
        """
    },
    {
        "text": "Yetkisiz kullanıcı admin paneline erişim sağladı.",
        "vulnerability": "A01:2021-Broken Access Control",
        "code_sample": """
        if not user.has_role('admin'):
            return "Access Denied"  # Missing role check
        """
    },
    {
        "text": "Kullanıcı, yetkisiz bir URL ile admin paneline erişti.",
        "vulnerability": "A01:2021-Broken Access Control",
        "code_sample": """
        @app.route('/admin')
        def admin_panel():
            return render_template('admin.html')  # No access control
        """
    },
    
    # A02:2021-Cryptographic Failures örnekleri
    {
        "text": "Şifreler düz metin olarak saklanıyor.",
        "vulnerability": "A02:2021-Cryptographic Failures",
        "code_sample": """
        def save_user_data(username, password):
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)", [username, password])  # Plain text password
        """
    },
    {
        "text": "Zayıf şifreleme algoritması kullanılıyor.",
        "vulnerability": "A02:2021-Cryptographic Failures",
        "code_sample": """
        def encrypt_data(data):
            return md5(data.encode()).hexdigest()  # Weak encryption
        """
    },
    {
        "text": "Şifreleme anahtarları düzenli olarak değiştirilmemekte.",
        "vulnerability": "A02:2021-Cryptographic Failures",
        "code_sample": """
        def get_encryption_key():
            return "my_secret_key"  # Hardcoded key
        """
    },
    
    # A03:2021-Injection örnekleri
    {
        "text": "SQL enjeksiyonu zafiyeti.",
        "vulnerability": "A03:2021-Injection",
        "code_sample": """
        def get_user(username):
            query = f"SELECT * FROM users WHERE username='{username}'"  # SQL Injection vulnerability
            return db.execute(query)
        """
    },
    {
        "text": "Komut enjeksiyonu zafiyeti.",
        "vulnerability": "A03:2021-Injection",
        "code_sample": """
        def execute_command(command):
            os.system(command)  # Command injection vulnerability
        """
    },
    {
        "text": "XSS zafiyeti.",
        "vulnerability": "A03:2021-Injection",
        "code_sample": """
        @app.route('/display')
        def display():
            return f"<p>{request.args.get('message')}</p>"  # XSS vulnerability
        """
    },
    
    # A04:2021-Insecure Direct Object References örnekleri
    {
        "text": "Yetkisiz dosya erişimi.",
        "vulnerability": "A04:2021-Insecure Direct Object References (IDOR)",
        "code_sample": """
        def get_file(file_id):
            return open(f"files/{file_id}")  # Insecure direct object reference
        """
    },
    {
        "text": "Kullanıcıların kendi dosyalarına erişim sağlaması.",
        "vulnerability": "A04:2021-Insecure Direct Object References (IDOR)",
        "code_sample": """
        def download_file(user_id, file_id):
            if user_id == get_file_owner(file_id):
                return open(f"files/{file_id}")  # IDOR vulnerability
        """
    },
    {
        "text": "Yetkisiz kullanıcıların dosya indirmesi.",
        "vulnerability": "A04:2021-Insecure Direct Object References (IDOR)",
        "code_sample": """
        def get_user_file(file_id):
            return open(f"user_files/{file_id}")  # IDOR vulnerability
        """
    },
    
    # Diğer zafiyet türleri için benzer şekilde örnekler ekleyin
]

class AdvancedVulnerabilityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def preprocess_text(self, text):
        # Temel metin temizleme
        doc = nlp(text.lower())
        
        # Gereksiz kelimeleri ve noktalama işaretlerini kaldır
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        # Kök kelimeleri bul
        lemmatized = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return " ".join(tokens) + " " + " ".join(lemmatized)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Metni önişle
        processed_text = self.preprocess_text(text)
        
        # Tokenize et
        encoding = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

class AdvancedVulnerabilityTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=2e-5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer ve learning rate scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
    
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
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return val_loss / len(self.val_loader), predictions, true_labels
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Accuracy hesapla
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.scheduler.step()
        epoch_accuracy = correct / total
        return total_loss / len(self.train_loader), epoch_accuracy

def plot_training_results(trainer):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(trainer.train_losses, label='Training Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(trainer.accuracies, label='Accuracy', color='green')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('static/training_results.png')
    plt.close()

def train_and_evaluate(trainer, epochs=5):
    best_accuracy = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Eğitim
        train_loss, train_accuracy = trainer.train_epoch()
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Değerlendirme
        val_loss, predictions, true_labels = trainer.evaluate()
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # En iyi modeli kaydet
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            trainer.model.save_pretrained('./best_model')
            print("Yeni en iyi model kaydedildi!")
        
        # Sınıflandırma raporu
        print("\nSınıflandırma Raporu:")
        print(classification_report(true_labels, predictions, zero_division=0))
        
        # Metrikleri kaydet
        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)
        trainer.accuracies.append(accuracy)
    
    # Eğitim sonuçlarını görselleştir
    plot_training_results(trainer)

if __name__ == "__main__":
    # Klasör yapısını oluştur
    os.makedirs('static', exist_ok=True)
    os.makedirs('vulnerability_model', exist_ok=True)
    
    # Model ve tokenizer yükleme
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(vulnerabilities_extended),
        problem_type="single_label_classification"
    )

    # Veri hazırlama
    texts = [f"{ex['text']}\n{ex['code_sample']}" for ex in training_examples]
    
    # Hata kontrolü ekleyelim
    valid_vulnerabilities = set(vulnerabilities_extended.keys())
    for example in training_examples:
        if example['vulnerability'] not in valid_vulnerabilities:
            raise ValueError(f"Geçersiz zafiyet türü: {example['vulnerability']}\n"
                           f"Geçerli zafiyetler: {valid_vulnerabilities}")
    
    labels = [list(vulnerabilities_extended.keys()).index(ex['vulnerability']) 
             for ex in training_examples]

    # Her sınıf için minimum örnek sayısını kontrol et
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    min_samples = min(label_counts)
    
    if min_samples < 3:
        raise ValueError(f"Her zafiyet türü için en az 3 örnek gerekli. "
                        f"En az örneğe sahip sınıf: {min_samples} örnek")

    # Veri bölme - stratify parametresini labels ile kullan
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, 
        test_size=0.3,  # %30 test verisi
        random_state=42,
        stratify=labels
    )

    # Dataset ve DataLoader oluşturma
    train_dataset = AdvancedVulnerabilityDataset(train_texts, train_labels, tokenizer)
    val_dataset = AdvancedVulnerabilityDataset(val_texts, val_labels, tokenizer)
    
    # Batch size'ı küçültelim
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Batch size'ı küçülttük
        shuffle=True,
        num_workers=0  # Windows'ta sorun çıkmaması için
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,  # Batch size'ı küçülttük
        num_workers=0  # Windows'ta sorun çıkmaması için
    )

    # Trainer oluşturma ve eğitim
    trainer = AdvancedVulnerabilityTrainer(model, train_loader, val_loader)
    train_and_evaluate(trainer, epochs=5)

    # Final modeli kaydet
    model.save_pretrained('./vulnerability_model')
    tokenizer.save_pretrained('./vulnerability_model')
    
    print("\nModel eğitimi tamamlandı ve kaydedildi.")

    print("Mevcut zafiyetler:", list(vulnerabilities_extended.keys()))
    print("\nEğitim örneklerindeki zafiyetler:")
    for ex in training_examples:
        print(f"- {ex['vulnerability']}")