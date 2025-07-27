import os
import re
import sqlite3
import time
import hashlib
import base64
import json
import uuid
import threading
import traceback
import logging
from datetime import datetime, timedelta
from io import BytesIO
from contextlib import contextmanager
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fingenius.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Image processing imports
try:
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    PIL_AVAILABLE = True
    CV2_AVAILABLE = True
    logger.info("✅ PIL and OpenCV loaded successfully")
except ImportError as e:
    PIL_AVAILABLE = False
    CV2_AVAILABLE = False
    logger.warning(f"⚠️ PIL/OpenCV not installed: {e}")

# OCR imports
try:
    import pytesseract
    # Test if tesseract binary is available
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
    logger.info("✅ Tesseract OCR loaded successfully")
except (ImportError, pytesseract.TesseractNotFoundError) as e:
    TESSERACT_AVAILABLE = False
    logger.warning(f"⚠️ Tesseract not available: {e}")

# Google Vision API (optional)
try:
    import importlib.util
    vision_spec = importlib.util.find_spec("google.cloud.vision")
    if vision_spec is not None:
        try:
            from google.cloud import vision
            VISION_API_AVAILABLE = bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
            if VISION_API_AVAILABLE:
                logger.info("✅ Google Vision API credentials found")
            else:
                logger.info("ℹ️ Google Vision API credentials not configured")
        except ImportError as e:
            VISION_API_AVAILABLE = False
            logger.info(f"ℹ️ Google Vision API import failed: {e}")
    else:
        VISION_API_AVAILABLE = False
        logger.info("ℹ️ Google Vision API not installed")
except ImportError:
    VISION_API_AVAILABLE = False
    logger.info("ℹ️ Google Vision API not installed")

# Twilio Integration
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
    logger.info("✅ Twilio library loaded")
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("⚠️ Twilio not installed")

# Constants
EXPENSE_CATEGORIES = [
    "Housing (Rent/Mortgage)",
    "Utilities (Electricity/Water)", 
    "Groceries",
    "Dining Out",
    "Transportation",
    "Healthcare",
    "Entertainment",
    "Education",
    "Personal Care",
    "Debt Payments",
    "Savings",
    "Investments",
    "Charity",
    "Miscellaneous"
]

INVESTMENT_TYPES = [
    "Stocks",
    "Bonds", 
    "Mutual Funds",
    "Real Estate",
    "Cryptocurrency",
    "Retirement Accounts",
    "Other"
]

RECURRENCE_PATTERNS = [
    "Daily",
    "Weekly",
    "Monthly", 
    "Quarterly",
    "Yearly"
]

# File upload constants
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
RECEIPTS_DIR = "receipts"

# Create receipts directory safely
try:
    os.makedirs(RECEIPTS_DIR, exist_ok=True)
    logger.info(f"📁 Receipts directory: {os.path.abspath(RECEIPTS_DIR)}")
except OSError as e:
    logger.error(f"❌ Could not create receipts directory: {e}")

# Security functions
def generate_salt():
    """Generate a random salt for password hashing"""
    return os.urandom(32).hex()

def hash_password(password, salt=None):
    """Hash password using SHA-256 with salt"""
    if salt is None:
        salt = generate_salt()
    password_hash = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
    return f"{password_hash}:{salt}"

def verify_password(password, hashed):
    """Verify password against hash"""
    try:
        if ':' not in hashed:
            logger.warning("Legacy password hash detected - please update")
            return False
        
        hash_part, salt = hashed.split(":", 1)
        computed_hash = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
        return computed_hash == hash_part
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def validate_phone_number(phone):
    """Validate phone number format - Pakistani mobile numbers"""
    if not phone:
        return False
    
    # Pakistan mobile numbers: +92 followed by 3 and then 9 digits
    pattern = r'^\+92[3][0-9]{9}$'
    return bool(re.match(pattern, phone))

def validate_password(password):
    """Validate password strength"""
    if not password:
        return False, "Password is required"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    return True, "Password is valid"

def format_currency(amount):
    """Format currency with proper PKR display"""
    if amount is None:
        return "0 PKR"
    return f"{int(amount):,} PKR"

def safe_file_extension(filename):
    """Get and validate file extension"""
    if not filename:
        return None
    
    ext = os.path.splitext(filename.lower())[1]
    return ext if ext in ALLOWED_IMAGE_EXTENSIONS else None

# ========== A) ENHANCED IMAGE PROCESSING ==========
class ImageProcessor:
    """Enhanced image preprocessing for better OCR results"""
    
    @staticmethod
    @contextmanager
    def open_image(image_path):
        """Context manager for safe image handling"""
        image = None
        try:
            image = Image.open(image_path)
            yield image
        finally:
            if image:
                image.close()
    
    @classmethod
    def preprocess_receipt_image(cls, image_path):
        """
        Preprocess receipt image for optimal OCR
        Returns: (processed_image_path, preprocessing_info)
        """
        try:
            if not PIL_AVAILABLE or not os.path.exists(image_path):
                return image_path, "No preprocessing available"
            
            with cls.open_image(image_path) as image:
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Enhance contrast and sharpness
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(2.0)
                
                # Convert to grayscale for better OCR
                image = image.convert('L')
                
                # Apply slight blur to reduce noise
                image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
                
                # OpenCV processing if available
                if CV2_AVAILABLE:
                    img_array = np.array(image)
                    
                    # Apply adaptive threshold
                    _, binary = cv2.threshold(
                        img_array, 0, 255, 
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    
                    # Clean up with morphological operations
                    kernel = np.ones((1, 1), np.uint8)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    
                    image = Image.fromarray(binary)
                
                # Save processed image
                processed_path = image_path.replace('.', '_processed.')
                image.save(processed_path, optimize=True, quality=95)
                
                return processed_path, "Enhanced: contrast, sharpness, thresholding applied"
                
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return image_path, f"Preprocessing failed: {str(e)}"
    
    @classmethod
    def process_receipt_image(cls, image_file, phone):
        """
        Complete receipt processing pipeline
        Returns: (success, status_message, extracted_data, image_preview_path)
        """
        try:
            if not phone:
                return False, "❌ Please sign in first", {}, None
            
            if not image_file:
                return False, "❌ No image uploaded", {}, None

            # Handle different input types from Gradio
            image_path = None
            if isinstance(image_file, str):
                image_path = image_file
            elif hasattr(image_file, 'name') and image_file.name:
                image_path = image_file.name
            else:
                return False, "❌ Invalid file format", {}, None

            # Validate file existence and extension
            if not os.path.exists(image_path):
                return False, "❌ File not found", {}, None
            
            file_ext = safe_file_extension(image_path)
            if not file_ext:
                return False, "❌ Invalid image format. Use JPG, PNG, or other supported formats", {}, None
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size > MAX_FILE_SIZE:
                return False, f"❌ File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB", {}, None
            
            # Generate secure filename
            timestamp = int(time.time())
            unique_id = uuid.uuid4().hex[:8]
            filename = f"receipt_{phone[-4:]}_{timestamp}_{unique_id}{file_ext}"
            save_path = os.path.abspath(os.path.join(RECEIPTS_DIR, filename))
            
            # Ensure the save path is within the receipts directory (security check)
            receipts_abs = os.path.abspath(RECEIPTS_DIR)
            if not save_path.startswith(receipts_abs):
                return False, "❌ Invalid file path", {}, None
            
            # Copy file safely with chunked reading
            try:
                with open(image_path, 'rb') as src, open(save_path, 'wb') as dst:
                    while True:
                        chunk = src.read(64 * 1024)  # 64KB chunks
                        if not chunk:
                            break
                        dst.write(chunk)
            except (IOError, OSError) as e:
                logger.error(f"File copy failed: {e}")
                return False, f"❌ File save failed: {str(e)}", {}, None
            
            logger.info(f"📄 Receipt saved: {save_path}")
            
            # Preprocess image
            processed_path, preprocessing_info = cls.preprocess_receipt_image(save_path)
            logger.info(f"🖼️ {preprocessing_info}")
            
            # Extract text using OCR
            raw_text, confidence, extracted_data = ocr_service.extract_text_from_receipt(processed_path)
            logger.info(f"🔍 OCR Confidence: {confidence:.1%}")
            
            # Auto-categorize expense
            if extracted_data.get('merchant'):
                suggested_category = db.auto_categorize_receipt(
                    phone, 
                    extracted_data['merchant'], 
                    extracted_data.get('total_amount', 0)
                )
                extracted_data['suggested_category'] = suggested_category
                logger.info(f"🏷️ Suggested category: {suggested_category}")
            
            # Save receipt data to database
            receipt_data = {
                'image_path': save_path,
                'processed_image_path': processed_path,
                'merchant': extracted_data.get('merchant', ''),
                'amount': extracted_data.get('total_amount', 0.0),
                'date': extracted_data.get('date', ''),
                'category': extracted_data.get('suggested_category', 'Miscellaneous'),
                'confidence': confidence,
                'raw_text': raw_text,
                'extracted_data': extracted_data,
                'is_validated': False
            }
            
            receipt_id = db.save_receipt(phone, receipt_data)
            if receipt_id:
                extracted_data['receipt_id'] = receipt_id
                logger.info(f"💾 Receipt saved to DB: {receipt_id}")
            
            # Generate status message
            status_msg = f"✅ Receipt processed! Confidence: {confidence:.1%}"
            if confidence < 0.7:
                status_msg += " ⚠️ Low confidence - please verify data"
            
            return True, status_msg, extracted_data, save_path
            
        except Exception as e:
            logger.error(f"Receipt processing error: {traceback.format_exc()}")
            return False, f"❌ Processing failed: {str(e)}", {}, None

# ========== B) ENHANCED OCR SERVICE ==========
class OCRService:
    """Enhanced OCR processing with multiple backends"""
    
    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE
        self.vision_api_available = VISION_API_AVAILABLE
        
        if self.vision_api_available:
            try:
                self.vision_client = vision.ImageAnnotatorClient()
                logger.info("✅ Google Vision API initialized")
            except Exception as e:
                logger.error(f"Google Vision API init failed: {e}")
                self.vision_api_available = False
    
    def extract_text_from_receipt(self, image_path):
        """
        Extract text from receipt using best available OCR service
        Returns: (raw_text, confidence_score, extracted_data)
        """
        try:
            if not os.path.exists(image_path):
                return "Image file not found", 0.0, self._create_empty_data()
            
            # Try Google Vision API first (more accurate)
            if self.vision_api_available:
                return self._extract_with_vision_api(image_path)
            
            # Fallback to Tesseract
            elif self.tesseract_available:
                return self._extract_with_tesseract(image_path)
            
            else:
                logger.warning("No OCR service available")
                return "OCR service not available", 0.0, self._create_empty_data()
                
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return f"OCR failed: {str(e)}", 0.0, self._create_empty_data()
    
    def _extract_with_vision_api(self, image_path):
        """Extract text using Google Vision API"""
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = self.vision_client.text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")
            
            texts = response.text_annotations
            
            if texts:
                raw_text = texts[0].description
                # Vision API doesn't provide confidence directly, estimate it
                confidence = 0.85  # Default high confidence for Vision API
                extracted_data = self._parse_receipt_text(raw_text)
                return raw_text, confidence, extracted_data
            else:
                return "No text detected by Vision API", 0.0, self._create_empty_data()
                
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return f"Vision API failed: {str(e)}", 0.0, self._create_empty_data()
    
    def _extract_with_tesseract(self, image_path):
        """Extract text using Tesseract OCR"""
        try:
            with ImageProcessor.open_image(image_path) as image:
                # Extract text with optimized config
                custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,$:/- '
                raw_text = pytesseract.image_to_string(image, config=custom_config)
                
                # Get confidence data
                data = pytesseract.image_to_data(
                    image, 
                    output_type=pytesseract.Output.DICT,
                    config=custom_config
                )
                
                # Calculate average confidence (filter out -1 values)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                avg_confidence = avg_confidence / 100.0  # Convert to 0-1 scale
                
                extracted_data = self._parse_receipt_text(raw_text)
                return raw_text, avg_confidence, extracted_data
                
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return f"Tesseract failed: {str(e)}", 0.0, self._create_empty_data()
    
    def _parse_receipt_text(self, raw_text):
        """Parse raw OCR text to extract structured data"""
        extracted_data = self._create_empty_data()
        
        if not raw_text or not raw_text.strip():
            return extracted_data
        
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        # Extract merchant name (first meaningful line)
        for line in lines:
            if len(line) > 2 and not re.match(r'^\d+[./\-]\d+', line):
                extracted_data['merchant'] = line[:50]  # Limit length
                break
        
        # Extract date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD
            r'\b\d{1,2}\s+\w{3,9}\s+\d{4}\b'       # DD Month YYYY
        ]
        
        for line in lines:
            for pattern in date_patterns:
                match = re.search(pattern, line)
                if match:
                    extracted_data['date'] = match.group().strip()
                    break
            if extracted_data['date']:
                break
        
        # Extract total amount (look for common patterns)
        amount_patterns = [
            r'(?:total|amount|sum|grand\s*total)[:\s]*(?:rs\.?|pkr)?\s*(\d+(?:[.,]\d{1,2})?)',
            r'(?:rs\.?|pkr)\s*(\d+(?:[.,]\d{1,2})?)',
            r'\b(\d+(?:[.,]\d{1,2})?)\s*(?:rs\.?|pkr)\b'
        ]
        
        amounts_found = []
        for line in lines:
            line_lower = line.lower()
            for pattern in amount_patterns:
                matches = re.finditer(pattern, line_lower)
                for match in matches:
                    try:
                        amount_str = match.group(1).replace(',', '.')
                        amount = float(amount_str)
                        if 1 <= amount <= 1000000:  # Reasonable range
                            amounts_found.append(amount)
                    except (ValueError, IndexError):
                        continue
        
        # Use the largest amount found (likely the total)
        if amounts_found:
            extracted_data['total_amount'] = max(amounts_found)
        
        # Extract line items
        line_items = []
        for line in lines:
            # Look for item-price patterns
            item_patterns = [
                r'(.+?)\s+(?:rs\.?|pkr)?\s*(\d+(?:[.,]\d{1,2})?)',
                r'(.+?)\s+(\d+(?:[.,]\d{1,2})?)\s*(?:rs\.?|pkr)?'
            ]
            
            for pattern in item_patterns:
                match = re.search(pattern, line.lower())
                if match and len(match.group(1).strip()) > 1:
                    try:
                        item_name = match.group(1).strip()[:30]  # Limit length
                        price_str = match.group(2).replace(',', '.')
                        price = float(price_str)
                        if 1 <= price <= 10000:  # Reasonable item price range
                            line_items.append([item_name, price])
                            if len(line_items) >= 10:  # Limit items
                                break
                    except (ValueError, IndexError):
                        continue
        
        extracted_data['line_items'] = line_items
        return extracted_data
    
    def _create_empty_data(self):
        """Create empty extracted data structure"""
        return {
            'merchant': '',
            'date': '',
            'total_amount': 0.0,
            'line_items': []
        }

# ========== C) ENHANCED DATABASE SERVICE ==========
class DatabaseService:
    """Enhanced database service with proper transaction handling"""
    
    def __init__(self, db_name='fingenius.db'):
        self.db_name = db_name
        self.db_lock = threading.RLock()  # Use RLock for nested calls
        self._initialize_db()
        logger.info(f"📊 Database initialized: {db_name}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_name, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _initialize_db(self):
        """Initialize database with proper schema"""
        with self.db_lock, self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table with proper constraints
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                phone TEXT PRIMARY KEY,
                name TEXT NOT NULL CHECK(length(name) > 0),
                password_hash TEXT NOT NULL,
                monthly_income INTEGER DEFAULT 0 CHECK(monthly_income >= 0),
                savings_goal INTEGER DEFAULT 0 CHECK(savings_goal >= 0),
                current_balance INTEGER DEFAULT 0,
                is_verified BOOLEAN DEFAULT FALSE,
                family_group TEXT DEFAULT NULL,
                last_balance_alert TIMESTAMP DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # Expenses table
            cursor.execute('''CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT NOT NULL,
                category TEXT NOT NULL CHECK(length(category) > 0),
                allocated INTEGER DEFAULT 0 CHECK(allocated >= 0),
                spent INTEGER DEFAULT 0 CHECK(spent >= 0),
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_recurring BOOLEAN DEFAULT FALSE,
                recurrence_pattern TEXT DEFAULT NULL,
                next_occurrence TIMESTAMP DEFAULT NULL,
                FOREIGN KEY(phone) REFERENCES users(phone) ON DELETE CASCADE
            )''')
            
            # Spending log table
            cursor.execute('''CREATE TABLE IF NOT EXISTS spending_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT NOT NULL,
                category TEXT NOT NULL,
                amount INTEGER NOT NULL CHECK(amount > 0),
                description TEXT DEFAULT '',
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                balance_after INTEGER NOT NULL,
                receipt_id TEXT DEFAULT NULL,
                FOREIGN KEY(phone) REFERENCES users(phone) ON DELETE CASCADE
            )''')
            
            # Receipts table
            cursor.execute('''CREATE TABLE IF NOT EXISTS receipts (
                receipt_id TEXT PRIMARY KEY,
                user_phone TEXT NOT NULL,
                image_path TEXT NOT NULL,
                processed_image_path TEXT,
                merchant TEXT DEFAULT '',
                amount REAL DEFAULT 0 CHECK(amount >= 0),
                receipt_date TEXT DEFAULT '',
                category TEXT DEFAULT 'Miscellaneous',
                ocr_confidence REAL DEFAULT 0 CHECK(ocr_confidence >= 0 AND ocr_confidence <= 1),
                raw_text TEXT DEFAULT '',
                extracted_data TEXT DEFAULT '{}',
                is_validated BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_phone) REFERENCES users(phone) ON DELETE CASCADE
            )''')
            
            # Other tables
            cursor.execute('''CREATE TABLE IF NOT EXISTS investments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT NOT NULL,
                type TEXT NOT NULL CHECK(length(type) > 0),
                name TEXT NOT NULL CHECK(length(name) > 0),
                amount INTEGER NOT NULL CHECK(amount > 0),
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT DEFAULT '',
                FOREIGN KEY(phone) REFERENCES users(phone) ON DELETE CASCADE
            )''')
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS family_groups (
                group_id TEXT PRIMARY KEY,
                name TEXT NOT NULL CHECK(length(name) > 0),
                admin_phone TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(admin_phone) REFERENCES users(phone)
            )''')
            
            # Create indexes for better performance
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_expenses_phone ON expenses(phone)',
                'CREATE INDEX IF NOT EXISTS idx_spending_log_phone ON spending_log(phone)',
                'CREATE INDEX IF NOT EXISTS idx_receipts_phone ON receipts(user_phone)',
                'CREATE INDEX IF NOT EXISTS idx_investments_phone ON investments(phone)',
                'CREATE INDEX IF NOT EXISTS idx_spending_log_date ON spending_log(date)',
                'CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(created_at)'
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            logger.info("✅ Database schema initialized with indexes")
    
    def create_user(self, phone, name, password):
        """Create new user with proper validation"""
        try:
            if not validate_phone_number(phone):
                return False, "Invalid phone number format"
            
            is_valid, msg = validate_password(password)
            if not is_valid:
                return False, msg
            
            password_hash = hash_password(password)
            
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO users (phone, name, password_hash, current_balance) 
                                 VALUES (?, ?, ?, ?)''', 
                              (phone, name.strip(), password_hash, 0))
                conn.commit()
                logger.info(f"👤 User created: {phone}")
                return True, "User created successfully"
                
        except sqlite3.IntegrityError:
            return False, "Phone number already registered"
        except Exception as e:
            logger.error(f"User creation error: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, phone, password):
        """Authenticate user with proper error handling"""
        try:
            if not validate_phone_number(phone) or not password:
                return None
            
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name, password_hash FROM users WHERE phone = ?', (phone,))
                result = cursor.fetchone()
                
                if result and verify_password(password, result['password_hash']):
                    logger.info(f"🔑 User authenticated: {phone}")
                    return result['name']
                
                return None
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def get_user(self, phone):
        """Get user data safely"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT name, monthly_income, savings_goal, family_group, current_balance 
                                 FROM users WHERE phone = ?''', (phone,))
                result = cursor.fetchone()
                return tuple(result) if result else None
        except Exception as e:
            logger.error(f"Get user error: {e}")
            return None
    
    def update_user_balance(self, phone, new_balance):
        """Update user balance with transaction"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE users SET current_balance = ? WHERE phone = ?', 
                              (new_balance, phone))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Balance update error: {e}")
            return False
    
    def get_current_balance(self, phone):
        """Get current balance safely"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT current_balance FROM users WHERE phone = ?', (phone,))
                result = cursor.fetchone()
                return result['current_balance'] if result else 0
        except Exception as e:
            logger.error(f"Get balance error: {e}")
            return 0
    
    def add_income(self, phone, amount, description="Income added"):
        """Add income with proper transaction handling"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get current balance
                cursor.execute('SELECT current_balance FROM users WHERE phone = ?', (phone,))
                result = cursor.fetchone()
                if not result:
                    return 0
                
                current_balance = result['current_balance']
                new_balance = current_balance + amount
                
                # Update balance and log transaction
                cursor.execute('UPDATE users SET current_balance = ? WHERE phone = ?',
                              (new_balance, phone))
                
                cursor.execute('''INSERT INTO spending_log 
                                 (phone, category, amount, description, balance_after)
                                 VALUES (?, ?, ?, ?, ?)''',
                              (phone, "Income", -amount, description, new_balance))
                
                conn.commit()
                logger.info(f"💰 Income added: {phone} - {amount}")
                return new_balance
                
        except Exception as e:
            logger.error(f"Add income error: {e}")
            return 0
    
    def update_financials(self, phone, income, savings):
        """Update financial goals with validation"""
        try:
            if income < 0 or savings < 0:
                return False
            
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''UPDATE users 
                                 SET monthly_income = ?, savings_goal = ?
                                 WHERE phone = ?''',
                              (income, savings, phone))
                conn.commit()
                logger.info(f"📊 Financials updated: {phone}")
                return True
                
        except Exception as e:
            logger.error(f"Update financials error: {e}")
            return False
    
    def get_expenses(self, phone, months_back=3):
        """Get expenses with proper date filtering"""
        try:
            end_date = datetime.now()
            start_date = end_date - relativedelta(months=months_back)
            
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT category, allocated, spent, date(date) as exp_date, is_recurring
                                 FROM expenses 
                                 WHERE phone = ? AND date BETWEEN ? AND ?
                                 ORDER BY allocated DESC''', 
                              (phone, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                
                return [tuple(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Get expenses error: {e}")
            return []
    
    def update_expense_allocations(self, phone, allocations):
        """Update expense allocations with transaction"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Clear existing allocations for non-recurring expenses
                cursor.execute('''DELETE FROM expenses 
                                 WHERE phone = ? AND allocated > 0 AND is_recurring = FALSE''', 
                              (phone,))
                
                # Insert new allocations
                for category, alloc in zip(EXPENSE_CATEGORIES, allocations):
                    if alloc > 0:
                        cursor.execute('''INSERT INTO expenses 
                                        (phone, category, allocated) 
                                        VALUES (?, ?, ?)''',
                                      (phone, category, alloc))
                
                conn.commit()
                logger.info(f"💼 Allocations updated: {phone}")
                return True
                
        except Exception as e:
            logger.error(f"Update allocations error: {e}")
            return False
    
    def record_expense(self, phone, category, amount, description="", is_recurring=False, recurrence_pattern=None, receipt_id=None):
        """Record expense with proper transaction handling"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get current balance
                cursor.execute('SELECT current_balance FROM users WHERE phone = ?', (phone,))
                result = cursor.fetchone()
                if not result:
                    return False, 0
                
                current_balance = result['current_balance']
                if current_balance < amount:
                    return False, current_balance
                
                new_balance = current_balance - amount
                
                # Update user balance
                cursor.execute('UPDATE users SET current_balance = ? WHERE phone = ?',
                              (new_balance, phone))
                
                # Log spending
                cursor.execute('''INSERT INTO spending_log 
                                 (phone, category, amount, description, balance_after, receipt_id)
                                 VALUES (?, ?, ?, ?, ?, ?)''',
                              (phone, category, amount, description, new_balance, receipt_id))
                
                # Handle recurring expenses
                if is_recurring and recurrence_pattern:
                    next_occurrence = self._calculate_next_occurrence(datetime.now(), recurrence_pattern)
                    cursor.execute('''INSERT INTO expenses 
                                    (phone, category, spent, is_recurring, recurrence_pattern, next_occurrence)
                                    VALUES (?, ?, ?, ?, ?, ?)''',
                                  (phone, category, amount, True, recurrence_pattern, next_occurrence))
                else:
                    # Update existing expense allocation
                    cursor.execute('''SELECT allocated, spent FROM expenses 
                                     WHERE phone = ? AND category = ? AND is_recurring = FALSE''',
                                  (phone, category))
                    expense_result = cursor.fetchone()
                    
                    if expense_result:
                        new_spent = expense_result['spent'] + amount
                        cursor.execute('''UPDATE expenses 
                                         SET spent = ? WHERE phone = ? AND category = ? AND is_recurring = FALSE''',
                                      (new_spent, phone, category))
                    else:
                        cursor.execute('''INSERT INTO expenses (phone, category, spent)
                                         VALUES (?, ?, ?)''',
                                      (phone, category, amount))
                
                conn.commit()
                logger.info(f"💸 Expense recorded: {phone} - {category} - {amount}")
                return True, new_balance
                
        except Exception as e:
            logger.error(f"Record expense error: {e}")
            return False, 0
    
    def _calculate_next_occurrence(self, current_date, pattern):
        """Calculate next occurrence for recurring expenses"""
        if pattern == "Daily":
            return current_date + timedelta(days=1)
        elif pattern == "Weekly":
            return current_date + timedelta(weeks=1)
        elif pattern == "Monthly":
            return current_date + relativedelta(months=1)
        elif pattern == "Quarterly":
            return current_date + relativedelta(months=3)
        elif pattern == "Yearly":
            return current_date + relativedelta(years=1)
        return current_date
    
    def record_investment(self, phone, inv_type, name, amount, notes):
        """Record investment with validation"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO investments 
                                 (phone, type, name, amount, notes)
                                 VALUES (?, ?, ?, ?, ?)''',
                              (phone, inv_type, name, amount, notes))
                conn.commit()
                logger.info(f"📈 Investment recorded: {phone} - {name}")
                return True
        except Exception as e:
            logger.error(f"Record investment error: {e}")
            return False
    
    def get_investments(self, phone):
        """Get user investments"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT type, name, amount, date(date) as inv_date, notes
                                 FROM investments 
                                 WHERE phone = ?
                                 ORDER BY date DESC''', (phone,))
                return [tuple(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Get investments error: {e}")
            return []
    
    def get_spending_log(self, phone, limit=50):
        """Get spending history"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT category, amount, description, date, balance_after
                                 FROM spending_log 
                                 WHERE phone = ? 
                                 ORDER BY date DESC 
                                 LIMIT ?''', (phone, limit))
                return [tuple(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Get spending log error: {e}")
            return []
    
    def save_receipt(self, phone, receipt_data):
        """Save receipt data to database"""
        try:
            receipt_id = f"REC-{phone[-4:]}-{uuid.uuid4().hex[:8]}"
            
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Safe JSON serialization
                try:
                    extracted_data_json = json.dumps(receipt_data.get('extracted_data', {}))
                except (TypeError, ValueError) as e:
                    logger.warning(f"JSON serialization warning: {e}")
                    extracted_data_json = "{}"
                
                cursor.execute('''INSERT INTO receipts 
                                 (receipt_id, user_phone, image_path, processed_image_path, 
                                  merchant, amount, receipt_date, category, ocr_confidence, 
                                  raw_text, extracted_data, is_validated)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                              (receipt_id, phone, 
                               receipt_data.get('image_path', ''),
                               receipt_data.get('processed_image_path', ''),
                               receipt_data.get('merchant', ''),
                               receipt_data.get('amount', 0.0),
                               receipt_data.get('date', ''),
                               receipt_data.get('category', ''),
                               receipt_data.get('confidence', 0.0),
                               receipt_data.get('raw_text', ''),
                               extracted_data_json,
                               receipt_data.get('is_validated', False)))
                
                conn.commit()
                logger.info(f"🧾 Receipt saved: {receipt_id}")
                return receipt_id
                
        except Exception as e:
            logger.error(f"Save receipt error: {e}")
            return None
    
    def get_receipts(self, phone, limit=20):
        """Get user receipts"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT receipt_id, merchant, amount, receipt_date, category, 
                                 ocr_confidence, is_validated, created_at
                                 FROM receipts 
                                 WHERE user_phone = ?
                                 ORDER BY created_at DESC 
                                 LIMIT ?''', (phone, limit))
                return [tuple(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Get receipts error: {e}")
            return []
    
    def update_receipt(self, receipt_id, updates):
        """Update receipt information safely"""
        if not updates:
            return False
        
        try:
            # Whitelist allowed columns for security
            allowed_columns = {
                'merchant': str,
                'amount': (int, float),
                'receipt_date': str,
                'category': str,
                'is_validated': bool,
                'raw_text': str
            }
            
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                
                set_clauses = []
                values = []
                
                for key, value in updates.items():
                    if key in allowed_columns:
                        # Type validation
                        expected_type = allowed_columns[key]
                        if isinstance(expected_type, tuple):
                            if not isinstance(value, expected_type):
                                continue
                        elif not isinstance(value, expected_type):
                            continue
                        
                        set_clauses.append(f"{key} = ?")
                        values.append(value)
                
                if set_clauses:
                    set_clause = ", ".join(set_clauses)
                    values.append(receipt_id)
                    
                    cursor.execute(f'UPDATE receipts SET {set_clause} WHERE receipt_id = ?', values)
                    conn.commit()
                    logger.info(f"📝 Receipt updated: {receipt_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Update receipt error: {e}")
            return False
    
    def auto_categorize_receipt(self, phone, merchant, amount):
        """Auto-categorize based on patterns and history"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check user's spending history for similar merchants
                cursor.execute('''SELECT category, COUNT(*) as count
                                 FROM spending_log 
                                 WHERE phone = ? AND (description LIKE ? OR description LIKE ?)
                                 GROUP BY category 
                                 ORDER BY count DESC 
                                 LIMIT 1''', 
                              (phone, f'%{merchant}%', f'%{merchant.split()[0] if merchant.split() else merchant}%'))
                
                result = cursor.fetchone()
                if result:
                    return result['category']
                
                # Fallback to keyword-based categorization
                return self._categorize_by_keywords(merchant)
                
        except Exception as e:
            logger.error(f"Auto categorize error: {e}")
            return "Miscellaneous"
    
    def _categorize_by_keywords(self, merchant):
        """Categorize based on merchant name keywords"""
        if not merchant:
            return "Miscellaneous"
        
        merchant_lower = merchant.lower()
        
        # Define keyword categories
        categories = {
            "Groceries": ['grocery', 'market', 'food', 'super', 'mart', 'store'],
            "Dining Out": ['restaurant', 'cafe', 'pizza', 'burger', 'hotel', 'dining'],
            "Transportation": ['gas', 'fuel', 'shell', 'bp', 'petrol', 'uber', 'taxi'],
            "Healthcare": ['pharmacy', 'medical', 'hospital', 'clinic', 'doctor'],
            "Utilities (Electricity/Water)": ['electric', 'water', 'utility', 'bill'],
            "Entertainment": ['cinema', 'movie', 'game', 'entertainment']
        }
        
        for category, keywords in categories.items():
            if any(keyword in merchant_lower for keyword in keywords):
                return category
        
        return "Miscellaneous"
    
    # Family group methods
    def create_family_group(self, group_name, admin_phone):
        """Create family group"""
        try:
            group_id = f"FG-{admin_phone[-4:]}-{uuid.uuid4().hex[:8]}"
            
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''INSERT INTO family_groups 
                                 (group_id, name, admin_phone) 
                                 VALUES (?, ?, ?)''',
                              (group_id, group_name, admin_phone))
                
                cursor.execute('''UPDATE users 
                                 SET family_group = ?
                                 WHERE phone = ?''',
                              (group_id, admin_phone))
                
                conn.commit()
                logger.info(f"👪 Family group created: {group_id}")
                return group_id
                
        except Exception as e:
            logger.error(f"Create family group error: {e}")
            return None
    
    def join_family_group(self, phone, group_id):
        """Join existing family group"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Verify group exists
                cursor.execute('SELECT name FROM family_groups WHERE group_id = ?', (group_id,))
                if not cursor.fetchone():
                    return False
                
                cursor.execute('UPDATE users SET family_group = ? WHERE phone = ?',
                              (group_id, phone))
                conn.commit()
                logger.info(f"👪 User joined family group: {phone} -> {group_id}")
                return True
                
        except Exception as e:
            logger.error(f"Join family group error: {e}")
            return False
    
    def get_family_group(self, group_id):
        """Get family group info"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name, admin_phone FROM family_groups WHERE group_id = ?', 
                              (group_id,))
                result = cursor.fetchone()
                return tuple(result) if result else None
        except Exception as e:
            logger.error(f"Get family group error: {e}")
            return None
    
    def get_family_members(self, group_id):
        """Get family group members"""
        try:
            with self.db_lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT phone, name FROM users WHERE family_group = ?', 
                              (group_id,))
                return [tuple(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Get family members error: {e}")
            return []

# ========== D) ENHANCED TWILIO SERVICE ==========
class TwilioWhatsAppService:
    """Enhanced Twilio WhatsApp service with better error handling"""
    
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID', 'your_account_sid_here')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN', 'your_auth_token_here')
        self.whatsapp_number = 'whatsapp:+14155238886'  # Twilio Sandbox
        
        self.enabled = False
        self.client = None
        
        if (self.account_sid != 'your_account_sid_here' and 
            self.auth_token != 'your_auth_token_here' and 
            TWILIO_AVAILABLE):
            
            try:
                self.client = Client(self.account_sid, self.auth_token)
                
                # Test connection
                account = self.client.api.accounts(self.account_sid).fetch()
                self.enabled = True
                logger.info(f"✅ Twilio initialized: {account.friendly_name}")
                
            except Exception as e:
                logger.error(f"❌ Twilio initialization failed: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ Twilio credentials not configured")
    
    def send_whatsapp(self, phone, message):
        """Send WhatsApp message with comprehensive error handling"""
        if not self.enabled or not self.client:
            logger.info(f"📱 [DEMO MODE] WhatsApp to {phone}: {message[:50]}...")
            return False
        
        try:
            # Format phone number
            if not phone.startswith('+'):
                phone = '+' + phone
            
            to_whatsapp = f"whatsapp:{phone}"
            
            # Send message
            twilio_message = self.client.messages.create(
                body=message[:1600],  # WhatsApp message limit
                from_=self.whatsapp_number,
                to=to_whatsapp
            )
            
            logger.info(f"✅ WhatsApp sent: {twilio_message.sid}")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Provide helpful error messages
            if "not a valid phone number" in error_msg:
                logger.error(f"❌ Invalid phone number format: {phone}")
            elif "unverified" in error_msg or "sandbox" in error_msg:
                logger.error(f"❌ WhatsApp not activated. User must send 'join catch-manner' to +14155238886")
            elif "forbidden" in error_msg:
                logger.error(f"❌ Twilio account issue. Check credentials and account status")
            else:
                logger.error(f"❌ WhatsApp send failed: {e}")
            
            return False

# ========== E) HELPER FUNCTIONS ==========
def generate_spending_chart(phone, months=3):
    """Generate spending chart with error handling"""
    try:
        expenses = db.get_expenses(phone, months)
        if not expenses:
            return None
        
        df = pd.DataFrame(expenses, columns=['Category', 'Allocated', 'Spent', 'Date', 'IsRecurring'])
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.strftime('%Y-%m')
        
        # Group by month and category
        monthly_data = df.groupby(['Month', 'Category'])['Spent'].sum().unstack(fill_value=0)
        
        # Create stacked bar chart
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        
        for i, category in enumerate(monthly_data.columns):
            fig.add_trace(go.Bar(
                x=monthly_data.index,
                y=monthly_data[category],
                name=category,
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>{category}</b><br>Month: %{{x}}<br>Amount: %{{y:,}} PKR<extra></extra>'
            ))
        
        fig.update_layout(
            barmode='stack',
            title=f'📊 Spending Trends (Last {months} Months)',
            xaxis_title='Month',
            yaxis_title='Amount (PKR)',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return None

def generate_balance_chart(phone):
    """Generate balance trend chart"""
    try:
        spending_log = db.get_spending_log(phone, 100)
        if not spending_log:
            return None
        
        df = pd.DataFrame(spending_log, columns=['Category', 'Amount', 'Description', 'Date', 'Balance'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Create line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Balance'],
            mode='lines+markers',
            name='Balance',
            line=dict(color='#00CC96', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x}<br><b>Balance:</b> %{y:,} PKR<extra></extra>',
            fill='tonexty' if len(df) > 1 else None,
            fillcolor='rgba(0, 204, 150, 0.1)'
        ))
        
        fig.update_layout(
            title='💰 Balance Trend Over Time',
            xaxis_title='Date',
            yaxis_title='Balance (PKR)',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Balance chart error: {e}")
        return None

# ========== F) INITIALIZE SERVICES ==========
try:
    db = DatabaseService()
    twilio = TwilioWhatsAppService()
    ocr_service = OCRService()
    logger.info("🚀 All services initialized successfully")
except Exception as e:
    logger.error(f"❌ Service initialization failed: {e}")
    raise

# ========== G) PAGE NAVIGATION FUNCTIONS ==========
def show_signin():
    """Show sign in page"""
    return [
        gr.update(visible=False),  # landing_page
        gr.update(visible=True),   # signin_page
        gr.update(visible=False),  # signup_page
        gr.update(visible=False),  # dashboard_page
        "",    # Clear signin inputs
        ""
    ]

def show_signup():
    """Show sign up page"""
    return [
        gr.update(visible=False),  # landing_page
        gr.update(visible=False),  # signin_page
        gr.update(visible=True),   # signup_page
        gr.update(visible=False),  # dashboard_page
        "",    # Clear signup inputs
        "",
        "",
        ""
    ]

def return_to_landing():
    """Return to landing page with preserved styling"""
    return [
        gr.update(visible=True),   # landing_page
        gr.update(visible=False),  # signin_page
        gr.update(visible=False),  # signup_page
        gr.update(visible=False),  # dashboard_page
        "",    # Clear welcome
        "<div class='balance-amount'>💰 0 PKR</div>"     # Clear balance
    ]

def show_dashboard(phone, name):
    """Show dashboard with user data"""
    try:
        user_data = db.get_user(phone)
        current_balance = user_data[4] if user_data else 0
        monthly_income = user_data[1] if user_data else 0
        savings_goal = user_data[2] if user_data else 0
        
        # Get expense data
        expenses = db.get_expenses(phone)
        formatted_expenses = []
        if expenses:
            for cat, alloc, spent, date, _ in expenses:
                remaining = alloc - spent
                formatted_expenses.append([
                    cat, alloc, spent, remaining, date.split()[0] if date else ""
                ])
        
        # Get investment data
        investments = db.get_investments(phone)
        formatted_investments = []
        if investments:
            for inv_type, inv_name, amount, date, notes in investments:
                formatted_investments.append([
                    inv_type, inv_name, amount, date.split()[0] if date else "", notes or ""
                ])
        
        # Get spending log
        spending_log = db.get_spending_log(phone, 10)
        formatted_spending_log = []
        if spending_log:
            for category, amount, description, date, balance_after in spending_log:
                desc_short = description[:50] + "..." if len(description) > 50 else description
                formatted_spending_log.append([
                    category, amount, desc_short,
                    date.split()[0] if date else "", balance_after
                ])
        
        # Get family info
        family_info = "No family group"
        family_members = []
        if user_data and user_data[3]:
            group_data = db.get_family_group(user_data[3])
            if group_data:
                family_info = f"Family Group: {group_data[0]} (Admin: {group_data[1]})"
                members = db.get_family_members(user_data[3])
                family_members = [[m[0], m[1]] for m in members]
        
        # Get receipt data
        receipts = db.get_receipts(phone)
        formatted_receipts = []
        if receipts:
            for receipt_id, merchant, amount, date, category, confidence, is_validated, created_at in receipts:
                status = "✅ Validated" if is_validated else "⏳ Pending"
                formatted_receipts.append([
                    receipt_id, merchant or "Unknown", format_currency(amount),
                    date or "N/A", category or "N/A", f"{confidence:.1%}",
                    status, created_at.split()[0] if created_at else ""
                ])
        
        # Prepare allocation inputs
        alloc_inputs = [0] * len(EXPENSE_CATEGORIES)
        if expenses:
            alloc_dict = {cat: alloc for cat, alloc, _, _, _ in expenses}
            alloc_inputs = [alloc_dict.get(cat, 0) for cat in EXPENSE_CATEGORIES]
        
        return [
            gr.update(visible=False),  # landing_page
            gr.update(visible=False),  # signin_page
            gr.update(visible=False),  # signup_page
            gr.update(visible=True),   # dashboard_page
            f"<div class='dashboard-welcome'>Welcome back, <strong>{name}</strong>! 👋</div>",  # welcome message
            f"<div class='balance-amount'>💰 {format_currency(current_balance)}</div>",  # balance display
            phone,  # current_user state
            monthly_income,  # income
            savings_goal,    # savings_goal
            *alloc_inputs,   # allocation inputs
            formatted_expenses,  # expense_table
            formatted_investments,  # investments_table
            formatted_spending_log,  # spending_log_table
            generate_spending_chart(phone),  # spending_chart
            generate_balance_chart(phone),   # balance_chart
            family_info,     # family_info
            family_members,  # family_members
            formatted_receipts  # receipts_table
        ]
        
    except Exception as e:
        logger.error(f"Show dashboard error: {e}")
        # Return safe default values
        empty_alloc = [0] * len(EXPENSE_CATEGORIES)
        return [
            gr.update(visible=False), gr.update(visible=False), 
            gr.update(visible=False), gr.update(visible=True),
            f"<div class='dashboard-welcome'>Welcome, <strong>{name}</strong>! (Error loading data)</div>", 
            "<div class='balance-amount'>💰 0 PKR</div>",
            phone, 0, 0, *empty_alloc, [], [], [], None, None, 
            "No family group", [], []
        ]

# ========== H) EVENT HANDLER FUNCTIONS ==========
def handle_signin(phone, password):
    """Handle user sign in"""
    try:
        if not phone or not password:
            # Return 32 values: status + 18 dashboard components + 13 more outputs
            return (
                "❌ Please fill all fields",  # signin_status
                gr.update(),  # landing_page  
                gr.update(),  # signin_page
                gr.update(),  # signup_page
                gr.update(),  # dashboard_page
                gr.update(),  # welcome_message
                gr.update(),  # balance_display
                gr.update(),  # current_user
                gr.update(),  # income
                gr.update(),  # savings_goal
                *[gr.update() for _ in range(14)],  # allocation_inputs (14 categories)
                gr.update(),  # expense_table
                gr.update(),  # investments_table
                gr.update(),  # spending_log_table
                gr.update(),  # spending_chart
                gr.update(),  # balance_chart
                gr.update(),  # family_info
                gr.update(),  # family_members
                gr.update()   # receipts_table
            )
        
        if not validate_phone_number(phone):
            return (
                "❌ Invalid phone format. Use +92XXXXXXXXXX",  # signin_status
                gr.update(),  # landing_page  
                gr.update(),  # signin_page
                gr.update(),  # signup_page
                gr.update(),  # dashboard_page
                gr.update(),  # welcome_message
                gr.update(),  # balance_display
                gr.update(),  # current_user
                gr.update(),  # income
                gr.update(),  # savings_goal
                *[gr.update() for _ in range(14)],  # allocation_inputs (14 categories)
                gr.update(),  # expense_table
                gr.update(),  # investments_table
                gr.update(),  # spending_log_table
                gr.update(),  # spending_chart
                gr.update(),  # balance_chart
                gr.update(),  # family_info
                gr.update(),  # family_members
                gr.update()   # receipts_table
            )
        
        user_name = db.authenticate_user(phone, password)
        
        if not user_name:
            return (
                "❌ Invalid phone number or password",  # signin_status
                gr.update(),  # landing_page  
                gr.update(),  # signin_page
                gr.update(),  # signup_page
                gr.update(),  # dashboard_page
                gr.update(),  # welcome_message
                gr.update(),  # balance_display
                gr.update(),  # current_user
                gr.update(),  # income
                gr.update(),  # savings_goal
                *[gr.update() for _ in range(14)],  # allocation_inputs (14 categories)
                gr.update(),  # expense_table
                gr.update(),  # investments_table
                gr.update(),  # spending_log_table
                gr.update(),  # spending_chart
                gr.update(),  # balance_chart
                gr.update(),  # family_info
                gr.update(),  # family_members
                gr.update()   # receipts_table
            )
        
        # Return successful login with dashboard data
        dashboard_data = show_dashboard(phone, user_name)
        return (f"✅ Signed in as {user_name}",) + tuple(dashboard_data)
        
    except Exception as e:
        logger.error(f"Sign in error: {e}")
        return (
            f"❌ Sign in failed: {str(e)}",  # signin_status
            gr.update(),  # landing_page  
            gr.update(),  # signin_page
            gr.update(),  # signup_page
            gr.update(),  # dashboard_page
            gr.update(),  # welcome_message
            gr.update(),  # balance_display
            gr.update(),  # current_user
            gr.update(),  # income
            gr.update(),  # savings_goal
            *[gr.update() for _ in range(14)],  # allocation_inputs (14 categories)
            gr.update(),  # expense_table
            gr.update(),  # investments_table
            gr.update(),  # spending_log_table
            gr.update(),  # spending_chart
            gr.update(),  # balance_chart
            gr.update(),  # family_info
            gr.update(),  # family_members
            gr.update()   # receipts_table
        )

def handle_signup(name, phone, password, confirm_password):
    """Handle user registration"""
    try:
        if not all([name, phone, password, confirm_password]):
            return "❌ Please fill all fields"
        
        if not validate_phone_number(phone):
            return "❌ Invalid phone format. Use +92XXXXXXXXXX"
        
        if password != confirm_password:
            return "❌ Passwords don't match"
        
        is_valid, password_msg = validate_password(password)
        if not is_valid:
            return f"❌ {password_msg}"
        
        success, msg = db.create_user(phone, name, password)
        if not success:
            return f"❌ {msg}"
        
        # Send welcome WhatsApp message
        welcome_msg = f"🏦 Welcome to FinGenius Pro, {name}! Your account has been created successfully. You can now track expenses, manage budgets, and receive instant financial alerts. Start by adding your first balance! 💰"
        whatsapp_sent = twilio.send_whatsapp(phone, welcome_msg)
        
        if whatsapp_sent:
            return "✅ Registration complete! Check WhatsApp for confirmation and sign in to continue."
        else:
            return "✅ Registration complete! WhatsApp alerts are not configured, but you can still use all features. Sign in to continue."
            
    except Exception as e:
        logger.error(f"Sign up error: {e}")
        return f"❌ Registration failed: {str(e)}"

def handle_add_balance(current_user, amount_val, description=""):
    """Handle adding balance to user account"""
    try:
        if not current_user:
            return "❌ Session expired. Please sign in again.", "<div class='balance-amount'>💰 0 PKR</div>"
        
        if not amount_val or amount_val <= 0:
            return "❌ Amount must be positive", "<div class='balance-amount'>💰 0 PKR</div>"
        
        new_balance = db.add_income(current_user, amount_val, description or "Balance added")
        
        user_data = db.get_user(current_user)
        if user_data:
            name = user_data[0]
            msg = f"💰 Balance Added - Hi {name}! Added: {format_currency(amount_val)}. New Balance: {format_currency(new_balance)}. Description: {description or 'Balance update'}"
            twilio.send_whatsapp(current_user, msg)
        
        return (
            f"✅ Added {format_currency(amount_val)} to balance!", 
            f"<div class='balance-amount'>💰 {format_currency(new_balance)}</div>"
        )
        
    except Exception as e:
        logger.error(f"Add balance error: {e}")
        return f"❌ Error adding balance: {str(e)}", "<div class='balance-amount'>💰 0 PKR</div>"

def handle_update_financials(current_user, income_val, savings_val):
    """Handle updating financial goals"""
    try:
        if not current_user:
            return "❌ Session expired. Please sign in again."
        
        if income_val < 0 or savings_val < 0:
            return "❌ Values cannot be negative"
        
        success = db.update_financials(current_user, income_val, savings_val)
        if not success:
            return "❌ Failed to update financial information"
        
        user_data = db.get_user(current_user)
        if user_data:
            name = user_data[0]
            msg = f"📊 Financial Goals Updated - Hi {name}! Monthly Income: {format_currency(income_val)}, Savings Goal: {format_currency(savings_val)}. Your budget planning is now ready! 🎯"
            twilio.send_whatsapp(current_user, msg)
        
        return f"✅ Updated! Monthly Income: {format_currency(income_val)}, Savings Goal: {format_currency(savings_val)}"
        
    except Exception as e:
        logger.error(f"Update financials error: {e}")
        return f"❌ Error updating financials: {str(e)}"

def handle_save_allocations(current_user, *allocations):
    """Handle saving budget allocations"""
    try:
        if not current_user:
            return "❌ Session expired. Please sign in again.", []
        
        if any(alloc < 0 for alloc in allocations):
            return "❌ Allocations cannot be negative", []
        
        total_alloc = sum(allocations)
        user_data = db.get_user(current_user)
        
        if not user_data:
            return "❌ User not found", []
        
        monthly_income = user_data[1]
        savings_goal = user_data[2]
        
        if total_alloc + savings_goal > monthly_income:
            return f"❌ Total allocations ({format_currency(total_alloc)}) + savings goal ({format_currency(savings_goal)}) exceed monthly income ({format_currency(monthly_income)})!", []
        
        success = db.update_expense_allocations(current_user, allocations)
        if not success:
            return "❌ Failed to save allocations", []
        
        name = user_data[0]
        msg = f"📋 Budget Allocated - Hi {name}! Your monthly budget has been set. Total allocated: {format_currency(total_alloc)}. Start tracking your expenses now! 💳"
        twilio.send_whatsapp(current_user, msg)
        
        # Get updated expenses
        expenses = db.get_expenses(current_user)
        formatted_expenses = []
        if expenses:
            for cat, alloc, spent, date, _ in expenses:
                remaining = alloc - spent
                formatted_expenses.append([
                    cat, alloc, spent, remaining, date.split()[0] if date else ""
                ])
        
        return "✅ Budget allocations saved!", formatted_expenses
        
    except Exception as e:
        logger.error(f"Save allocations error: {e}")
        return f"❌ Error saving allocations: {str(e)}", []

def handle_record_expense(current_user, category, amount, description="", is_recurring=False, recurrence_pattern=None):
    """Handle recording an expense"""
    try:
        if not current_user:
            return "❌ Session expired. Please sign in again.", "<div class='balance-amount'>💰 0 PKR</div>", [], []
        
        if not amount or amount <= 0:
            return "❌ Amount must be positive", "<div class='balance-amount'>💰 0 PKR</div>", [], []
        
        if not category:
            return "❌ Please select a category", "<div class='balance-amount'>💰 0 PKR</div>", [], []
        
        current_balance = db.get_current_balance(current_user)
        if current_balance < amount:
            return f"❌ Insufficient balance. Current: {format_currency(current_balance)}", f"<div class='balance-amount'>💰 {format_currency(current_balance)}</div>", [], []
        
        success, new_balance = db.record_expense(current_user, category, amount, description, is_recurring, recurrence_pattern)
        
        if not success:
            return "❌ Failed to record expense", f"<div class='balance-amount'>💰 {format_currency(current_balance)}</div>", [], []
        
        user_data = db.get_user(current_user)
        name = user_data[0] if user_data else "User"
        
        msg = f"💸 Expense Recorded - Hi {name}! Category: {category}, Amount: {format_currency(amount)}, Remaining Balance: {format_currency(new_balance)}"
        if description:
            msg += f", Note: {description}"
        if is_recurring:
            msg += f" (Recurring: {recurrence_pattern})"
        twilio.send_whatsapp(current_user, msg)
        
        # Get updated data
        expenses = db.get_expenses(current_user)
        formatted_expenses = []
        if expenses:
            for cat, alloc, spent, date, _ in expenses:
                remaining = alloc - spent
                formatted_expenses.append([
                    cat, alloc, spent, remaining, date.split()[0] if date else ""
                ])
        
        spending_log = db.get_spending_log(current_user, 10)
        formatted_spending_log = []
        if spending_log:
            for cat, amt, desc, date, balance_after in spending_log:
                desc_short = desc[:50] + "..." if len(desc) > 50 else desc
                formatted_spending_log.append([
                    cat, amt, desc_short, date.split()[0] if date else "", balance_after
                ])
        
        status_msg = f"✅ Recorded {format_currency(amount)} for {category}"
        balance_html = f"<div class='balance-amount'>💰 {format_currency(new_balance)}</div>"
        
        return status_msg, balance_html, formatted_expenses, formatted_spending_log
        
    except Exception as e:
        logger.error(f"Record expense error: {e}")
        current_balance = db.get_current_balance(current_user) if current_user else 0
        return f"❌ Error recording expense: {str(e)}", f"<div class='balance-amount'>💰 {format_currency(current_balance)}</div>", [], []

def handle_add_investment(current_user, inv_type, name, amount, notes):
    """Handle adding an investment"""
    try:
        if not current_user:
            return "❌ Session expired. Please sign in again.", "<div class='balance-amount'>💰 0 PKR</div>", []
        
        if not amount or amount <= 0:
            return "❌ Amount must be positive", "<div class='balance-amount'>💰 0 PKR</div>", []
        
        if not inv_type or not name:
            return "❌ Please fill investment type and name", "<div class='balance-amount'>💰 0 PKR</div>", []
        
        current_balance = db.get_current_balance(current_user)
        if current_balance < amount:
            return f"❌ Insufficient balance. Current: {format_currency(current_balance)}", f"<div class='balance-amount'>💰 {format_currency(current_balance)}</div>", []
        
        # Record as expense first (deduct from balance)
        success, new_balance = db.record_expense(current_user, "Investments", amount, f"Investment: {name}")
        
        if not success:
            return "❌ Failed to process investment", f"<div class='balance-amount'>💰 {format_currency(current_balance)}</div>", []
        
        # Record investment
        inv_success = db.record_investment(current_user, inv_type, name, amount, notes)
        
        if not inv_success:
            return "❌ Investment recorded but failed to save details", f"<div class='balance-amount'>💰 {format_currency(new_balance)}</div>", []
        
        user_data = db.get_user(current_user)
        if user_data:
            user_name = user_data[0]
            msg = f"📈 Investment Added - Hi {user_name}! Type: {inv_type}, Name: {name}, Amount: {format_currency(amount)}, Remaining Balance: {format_currency(new_balance)}"
            if notes:
                msg += f", Notes: {notes}"
            twilio.send_whatsapp(current_user, msg)
        
        # Get updated investments
        investments = db.get_investments(current_user)
        formatted_investments = []
        if investments:
            for inv_type_db, name_db, amount_db, date, notes_db in investments:
                formatted_investments.append([
                    inv_type_db, name_db, amount_db, date.split()[0] if date else "", notes_db or ""
                ])
        
        balance_html = f"<div class='balance-amount'>💰 {format_currency(new_balance)}</div>"
        
        return f"✅ Added investment: {name} ({format_currency(amount)})", balance_html, formatted_investments
        
    except Exception as e:
        logger.error(f"Add investment error: {e}")
        current_balance = db.get_current_balance(current_user) if current_user else 0
        return f"❌ Error adding investment: {str(e)}", f"<div class='balance-amount'>💰 {format_currency(current_balance)}</div>", []

def handle_create_family_group(current_user, group_name):
    """Handle creating a family group"""
    try:
        if not current_user or not group_name:
            return "❌ Group name required", "", []
        
        group_id = db.create_family_group(group_name, current_user)
        if not group_id:
            return "❌ Failed to create group", "", []
        
        user_data = db.get_user(current_user)
        if user_data:
            name = user_data[0]
            msg = f"👪 Family Group Created - Hi {name}! You've created '{group_name}' (ID: {group_id}). Share this ID with family members to join. Manage finances together! 🏠"
            twilio.send_whatsapp(current_user, msg)
        
        # Get current user as first member
        family_members = [[current_user, user_data[0] if user_data else "You"]]
        
        return (
            f"✅ Created group: {group_name} (ID: {group_id})", 
            f"Family Group: {group_name} (Admin: {current_user})", 
            family_members
        )
        
    except Exception as e:
        logger.error(f"Create family group error: {e}")
        return f"❌ Error creating group: {str(e)}", "", []

def handle_join_family_group(current_user, group_id):
    """Handle joining a family group"""
    try:
        if not current_user or not group_id:
            return "❌ Group ID required", "", []
        
        success = db.join_family_group(current_user, group_id)
        if not success:
            return "❌ Failed to join group. Check group ID.", "", []
        
        group_data = db.get_family_group(group_id)
        if not group_data:
            return "❌ Group not found", "", []
        
        user_data = db.get_user(current_user)
        if user_data:
            name = user_data[0]
            msg = f"👪 Joined Family Group - Hi {name}! You've joined '{group_data[0]}'. Start collaborating on family finances together! 🤝"
            twilio.send_whatsapp(current_user, msg)
        
        members = db.get_family_members(group_id)
        member_list = [[m[0], m[1]] for m in members]
        
        return (
            f"✅ Joined group: {group_data[0]}", 
            f"Family Group: {group_data[0]} (Admin: {group_data[1]})", 
            member_list
        )
        
    except Exception as e:
        logger.error(f"Join family group error: {e}")
        return f"❌ Error joining group: {str(e)}", "", []

def handle_update_charts(current_user, months_history):
    """Handle updating analytics charts"""
    try:
        if not current_user:
            return None, None
        
        spending_chart = generate_spending_chart(current_user, months_history)
        balance_chart = generate_balance_chart(current_user)
        
        return spending_chart, balance_chart
        
    except Exception as e:
        logger.error(f"Update charts error: {e}")
        return None, None

# Receipt processing event handlers
def handle_receipt_upload(image_file, current_user):
    """Handle receipt image upload and processing"""
    try:
        if not current_user:
            return "❌ Please sign in first", {}, "", "", "", [], None, ""
        
        if not image_file:
            return "❌ Please upload an image", {}, "", "", "", [], None, ""
        
        # Process the receipt
        success, status, extracted_data, image_path = ImageProcessor.process_receipt_image(image_file, current_user)
        
        if not success:
            return status, {}, "", "", "", [], None, ""
        
        # Prepare UI updates
        merchant = extracted_data.get('merchant', '')
        amount = extracted_data.get('total_amount', 0.0)
        date = extracted_data.get('date', '')
        category = extracted_data.get('suggested_category', 'Miscellaneous')
        line_items = extracted_data.get('line_items', [])
        
        # Create image preview if available
        image_preview = None
        if image_path and os.path.exists(image_path):
            try:
                with ImageProcessor.open_image(image_path) as img:
                    img.thumbnail((400, 600))
                    preview_path = image_path.replace('.', '_preview.')
                    img.save(preview_path)
                    image_preview = preview_path
            except Exception as e:
                logger.warning(f"Preview generation failed: {e}")
        
        receipt_data = {
            "receipt_id": extracted_data.get('receipt_id', ''), 
            "confidence": extracted_data.get('confidence', 0.0)
        }
        
        return (
            status,
            receipt_data,
            merchant,
            amount,
            date,
            line_items,
            image_preview,
            category
        )
        
    except Exception as e:
        logger.error(f"Receipt upload error: {e}")
        return f"❌ Upload failed: {str(e)}", {}, "", "", "", [], None, ""

def handle_receipt_save(current_user, receipt_data, merchant, amount, date, category, line_items_data):
    """Save validated receipt as expense"""
    try:
        if not current_user or not receipt_data:
            return "❌ No receipt data to save", "<div class='balance-amount'>💰 0 PKR</div>", [], []
        
        receipt_id = receipt_data.get('receipt_id')
        if not receipt_id:
            return "❌ Invalid receipt data", "<div class='balance-amount'>💰 0 PKR</div>", [], []
        
        if not merchant.strip():
            return "❌ Merchant name is required", "<div class='balance-amount'>💰 0 PKR</div>", [], []
        
        if amount <= 0:
            return "❌ Amount must be positive", "<div class='balance-amount'>💰 0 PKR</div>", [], []
        
        # Check balance
        current_balance = db.get_current_balance(current_user)
        if current_balance < amount:
            return f"❌ Insufficient balance. Current: {format_currency(current_balance)}", f"<div class='balance-amount'>💰 {format_currency(current_balance)}</div>", [], []
        
        # Update receipt in database
        receipt_updates = {
            'merchant': merchant.strip(),
            'amount': amount,
            'receipt_date': date.strip(),
            'category': category,
            'is_validated': True
        }
        db.update_receipt(receipt_id, receipt_updates)
        
        # Record as expense
        description = f"Receipt: {merchant}"
        if date:
            description += f" ({date})"
        
        success, new_balance = db.record_expense(current_user, category, amount, description, receipt_id=receipt_id)
        
        if not success:
            return "❌ Failed to record expense", f"<div class='balance-amount'>💰 {format_currency(current_balance)}</div>", [], []
        
        # Send WhatsApp confirmation
        user_data = db.get_user(current_user)
        name = user_data[0] if user_data else "User"
        
        msg = f"🧾 Receipt Expense - Hi {name}! Merchant: {merchant}, Amount: {format_currency(amount)}, Category: {category}, Remaining Balance: {format_currency(new_balance)}"
        twilio.send_whatsapp(current_user, msg)
        
        # Get updated data for UI
        expenses = db.get_expenses(current_user)
        formatted_expenses = []
        if expenses:
            for cat, alloc, spent, exp_date, _ in expenses:
                remaining = alloc - spent
                formatted_expenses.append([
                    cat, alloc, spent, remaining, exp_date.split()[0] if exp_date else ""
                ])
        
        spending_log = db.get_spending_log(current_user, 10)
        formatted_spending_log = []
        if spending_log:
            for cat, amt, desc, log_date, balance_after in spending_log:
                desc_short = desc[:50] + "..." if len(desc) > 50 else desc
                formatted_spending_log.append([
                    cat, amt, desc_short, log_date.split()[0] if log_date else "", balance_after
                ])
        
        status_msg = f"✅ Receipt saved! Recorded {format_currency(amount)} for {category}"
        balance_html = f"<div class='balance-amount'>💰 {format_currency(new_balance)}</div>"
        
        return status_msg, balance_html, formatted_expenses, formatted_spending_log
        
    except Exception as e:
        logger.error(f"Receipt save error: {e}")
        current_balance = db.get_current_balance(current_user) if current_user else 0
        return f"❌ Error saving receipt: {str(e)}", f"<div class='balance-amount'>💰 {format_currency(current_balance)}</div>", [], []

# ========== I) ENHANCED CUSTOM CSS ==========
custom_css = """
/* Enhanced CSS for better UI/UX with fixed sizing issues */
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #ff6b6b;
    --success-color: #48bb78;
    --warning-color: #ed8936;
    --error-color: #e53e3e;
    --text-primary: #2d3748;
    --text-secondary: #4a5568;
    --bg-light: #f7fafc;
    --bg-card: #ffffff;
    --border-color: #e2e8f0;
    --shadow-light: 0 4px 6px rgba(0, 0, 0, 0.05);
    --shadow-medium: 0 10px 25px rgba(0, 0, 0, 0.1);
    --shadow-heavy: 0 20px 40px rgba(0, 0, 0, 0.15);
    --border-radius: 15px;
    --border-radius-small: 8px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 1rem !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    min-height: 100vh !important;
}

/* Fixed Landing Page Styling */
.landing-hero {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    min-height: 85vh;
    padding: 4rem 2rem;
    color: white;
    text-align: center;
    border-radius: var(--border-radius);
    margin: 1rem 0;
    box-shadow: var(--shadow-heavy);
    position: relative;
    overflow: hidden;
}

.landing-hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%, rgba(255,255,255,0.05) 100%);
    pointer-events: none;
}

.hero-title {
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    margin-bottom: 1.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    background: linear-gradient(45deg, #fff, #f0f8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    z-index: 1;
}

.hero-subtitle {
    font-size: clamp(1.1rem, 2.5vw, 1.6rem);
    margin-bottom: 3rem;
    opacity: 0.95;
    font-weight: 300;
    line-height: 1.6;
    position: relative;
    z-index: 1;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin: 3rem 0;
    position: relative;
    z-index: 1;
}

.feature-card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
    border-radius: var(--border-radius);
    padding: 2.5rem 2rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    transition: left 0.8s;
}

.feature-card:hover::before {
    left: 100%;
}

.feature-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    border-color: rgba(255,255,255,0.3);
}

.feature-icon {
    font-size: 3.5rem;
    margin-bottom: 1.5rem;
    display: block;
    transform: scale(1);
    transition: var(--transition);
}

.feature-card:hover .feature-icon {
    transform: scale(1.1) rotate(5deg);
}

.feature-card h3 {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: white;
}

.feature-card p {
    opacity: 0.9;
    line-height: 1.6;
    font-size: 1rem;
}

/* Fixed Auth Container Styling */
.auth-container {
    max-width: 480px;
    margin: 2rem auto;
    background: var(--bg-card);
    border-radius: var(--border-radius);
    padding: 3rem 2.5rem;
    box-shadow: var(--shadow-heavy);
    border: 1px solid var(--border-color);
    position: relative;
    backdrop-filter: blur(10px);
}

.auth-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    border-radius: var(--border-radius) var(--border-radius) 0 0;
}

/* WhatsApp Setup Enhanced */
.whatsapp-setup {
    background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
    color: white;
    padding: 2.5rem;
    border-radius: var(--border-radius);
    margin: 2rem 0;
    text-align: center;
    box-shadow: 0 15px 35px rgba(37, 211, 102, 0.3);
    position: relative;
    overflow: hidden;
}

.whatsapp-setup::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.05), transparent);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

.whatsapp-steps {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(15px);
    border-radius: var(--border-radius-small);
    padding: 2rem;
    margin: 1.5rem 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
    text-align: left;
    position: relative;
    z-index: 1;
}

.whatsapp-steps h4 {
    color: white;
    font-weight: 600;
    margin-bottom: 1rem;
    font-size: 1.2rem;
}

.phone-highlight, .code-highlight {
    background: rgba(255, 255, 255, 0.25);
    padding: 0.8rem 1.2rem;
    border-radius: var(--border-radius-small);
    font-family: 'SF Mono', 'Monaco', 'Cascadia Code', 'Roboto Mono', monospace;
    font-size: 1.1rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.8rem 0;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}

.code-highlight {
    border-left: 4px solid rgba(255, 255, 255, 0.5);
}

/* Enhanced Dashboard Styling */
.dashboard-header {
    background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-secondary) 100%);
    color: white;
    padding: 2.5rem;
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    text-align: center;
    font-size: 1.6rem;
    box-shadow: var(--shadow-medium);
    position: relative;
    overflow: hidden;
}

.dashboard-welcome {
    font-size: 1.8rem;
    font-weight: 300;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}

.dashboard-welcome strong {
    font-weight: 600;
    color: #ffd700;
}

.balance-card {
    background: linear-gradient(135deg, var(--success-color) 0%, #38a169 100%);
    color: white;
    padding: 2.5rem;
    border-radius: var(--border-radius);
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-medium);
    position: relative;
    overflow: hidden;
}

.balance-card::after {
    content: '💰';
    position: absolute;
    top: -20px;
    right: -20px;
    font-size: 8rem;
    opacity: 0.1;
    pointer-events: none;
}

.balance-amount {
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    margin: 1.5rem 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    background: linear-gradient(45deg, #fff, #f0fff0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Enhanced Button Styling with Equal Sizes */
.primary-btn, .secondary-btn {
    border: none !important;
    border-radius: 25px !important;
    padding: 1rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: white !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    min-width: 180px !important;
    height: 48px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-decoration: none !important;
    box-sizing: border-box !important;
}

.primary-btn {
    background: linear-gradient(45deg, var(--accent-color), #ee5a24) !important;
    box-shadow: 0 4px 15px rgba(238, 90, 36, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(238, 90, 36, 0.6) !important;
}

.secondary-btn {
    background: linear-gradient(45deg, #74b9ff, #0984e3) !important;
    box-shadow: 0 4px 15px rgba(116, 185, 255, 0.4) !important;
}

.secondary-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(116, 185, 255, 0.6) !important;
}

.primary-btn::before, .secondary-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.primary-btn:hover::before, .secondary-btn:hover::before {
    left: 100%;
}

/* Enhanced Table Styling */
.dataframe {
    border-radius: var(--border-radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-medium) !important;
    border: 1px solid var(--border-color) !important;
    background: var(--bg-card) !important;
    margin: 1rem 0 !important;
}

.dataframe th {
    background: linear-gradient(135deg, var(--bg-light) 0%, #edf2f7 100%) !important;
    font-weight: 600 !important;
    padding: 1.2rem 1rem !important;
    border-bottom: 2px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.dataframe td {
    padding: 1rem !important;
    border-bottom: 1px solid #f1f5f9 !important;
    color: var(--text-secondary) !important;
    transition: background-color 0.2s ease !important;
}

.dataframe tr:hover td {
    background-color: rgba(102, 126, 234, 0.05) !important;
}

/* Enhanced Tab Styling */
.tab-nav .tab-nav {
    border-radius: var(--border-radius) !important;
    background: var(--bg-card) !important;
    box-shadow: var(--shadow-light) !important;
    border: 1px solid var(--border-color) !important;
    overflow: hidden !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    padding: 1rem 1.5rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    transition: var(--transition) !important;
    position: relative !important;
}

.tab-nav button:hover {
    background: rgba(102, 126, 234, 0.05) !important;
    color: var(--primary-color) !important;
}

.tab-nav button.selected {
    background: var(--primary-color) !important;
    color: white !important;
}

.tab-nav button.selected::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--accent-color);
}

/* Enhanced Form Styling */
.gr-form {
    background: var(--bg-card) !important;
    border-radius: var(--border-radius) !important;
    padding: 2rem !important;
    box-shadow: var(--shadow-light) !important;
    border: 1px solid var(--border-color) !important;
    margin: 1rem 0 !important;
}

.gr-form label {
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-bottom: 0.5rem !important;
    display: block !important;
}

.gr-form input, .gr-form textarea, .gr-form select {
    border: 2px solid var(--border-color) !important;
    border-radius: var(--border-radius-small) !important;
    padding: 0.8rem 1rem !important;
    font-size: 1rem !important;
    transition: var(--transition) !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

.gr-form input:focus, .gr-form textarea:focus, .gr-form select:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    outline: none !important;
}

/* Status Messages */
.status-success {
    color: var(--success-color) !important;
    background: rgba(72, 187, 120, 0.1) !important;
    border: 1px solid rgba(72, 187, 120, 0.2) !important;
    padding: 1rem !important;
    border-radius: var(--border-radius-small) !important;
    font-weight: 600 !important;
    margin: 1rem 0 !important;
}

.status-error {
    color: var(--error-color) !important;
    background: rgba(229, 62, 62, 0.1) !important;
    border: 1px solid rgba(229, 62, 62, 0.2) !important;
    padding: 1rem !important;
    border-radius: var(--border-radius-small) !important;
    font-weight: 600 !important;
    margin: 1rem 0 !important;
}

/* Enhanced Cards and Sections */
.info-card {
    background: #000000;
    border-radius: var(--border-radius);
    padding: 2rem;
    box-shadow: var(--shadow-light);
    border: 1px solid var(--border-color);
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}

.info-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
}

.info-card h3 {
    color: var(--text-primary);
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.info-card p {
    color: var(--text-secondary);
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* Responsive Design Enhancements */
@media (max-width: 768px) {
    .gradio-container {
        padding: 0.5rem !important;
    }
    
    .hero-title {
        font-size: 2.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
    }
    
    .features-grid {
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }
    
    .auth-container {
        margin: 1rem;
        padding: 2rem 1.5rem;
    }
    
    .balance-amount {
        font-size: 2.2rem;
    }
    
    .primary-btn, .secondary-btn {
        min-width: 150px !important;
        font-size: 1rem !important;
        padding: 0.9rem 1.5rem !important;
    }
    
    .dashboard-welcome {
        font-size: 1.4rem;
    }
    
    .feature-card {
        padding: 2rem 1.5rem;
    }
    
    .whatsapp-setup {
        padding: 2rem 1.5rem;
    }
    
    .whatsapp-steps {
        padding: 1.5rem;
    }
}

@media (max-width: 480px) {
    .hero-title {
        font-size: 2rem;
    }
    
    .balance-amount {
        font-size: 2rem;
    }
    
    .auth-container {
        padding: 2rem 1rem;
    }
    
    .primary-btn, .secondary-btn {
        min-width: 130px !important;
        padding: 0.8rem 1.2rem !important;
    }
}

/* Loading States */
.loading {
    opacity: 0.6;
    pointer-events: none;
    position: relative;
}

.loading::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 20px;
    height: 20px;
    margin: -10px 0 0 -10px;
    border: 2px solid transparent;
    border-top-color: var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Animations */
@keyframes fadeIn {
    from { 
        opacity: 0; 
        transform: translateY(30px); 
    }
    to { 
        opacity: 1; 
        transform: translateY(0); 
    }
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.fade-in {
    animation: fadeIn 0.6s ease-out;
}

.slide-in {
    animation: slideIn 0.6s ease-out;
}

/* Enhanced Gradient Backgrounds */
.gradient-bg-1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.gradient-bg-2 {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.gradient-bg-3 {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.gradient-bg-4 {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

/* Chart Container Enhancements */
.plotly-graph-div {
    border-radius: var(--border-radius) !important;
    box-shadow: var(--shadow-light) !important;
    border: 1px solid var(--border-color) !important;
    background: var(--bg-card) !important;
    margin: 1rem 0 !important;
}

/* File Upload Styling */
.file-upload {
    border: 2px dashed var(--border-color) !important;
    border-radius: var(--border-radius) !important;
    padding: 2rem !important;
    text-align: center !important;
    background: var(--bg-light) !important;
    transition: var(--transition) !important;
}

.file-upload:hover {
    border-color: var(--primary-color) !important;
    background: rgba(102, 126, 234, 0.05) !important;
}

/* Enhanced Accordion Styling */
.gr-accordion {
    border-radius: var(--border-radius) !important;
    border: 1px solid var(--border-color) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-light) !important;
}

.gr-accordion summary {
    background: var(--bg-light) !important;
    padding: 1rem 1.5rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
}

.gr-accordion summary:hover {
    background: rgba(102, 126, 234, 0.05) !important;
}

.gr-accordion[open] summary {
    background: var(--primary-color) !important;
    color: white !important;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-light);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
    transition: var(--transition);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-secondary);
}
"""

# ========== J) MAIN GRADIO INTERFACE ==========
with gr.Blocks(title="FinGenius Pro", theme=gr.themes.Soft(), css=custom_css) as demo:
    # State to track current user
    current_user = gr.State("")
    receipt_data = gr.State({})
    
    # ===== LANDING PAGE =====
    with gr.Column(visible=True, elem_classes="fade-in") as landing_page:
        gr.HTML("""
            <div class="landing-hero">
                <div class="hero-title">🏦 FinGenius Pro</div>
                <div class="hero-subtitle">Your Complete Personal Finance Manager with Smart AI Alerts</div>
                
                <div class="features-grid">
                    <div class="feature-card">
                        <div class="feature-icon">💰</div>
                        <h3>Smart Balance Tracking</h3>
                        <p>Real-time balance monitoring with intelligent spending alerts and comprehensive financial insights</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📱</div>
                        <h3>WhatsApp Integration</h3>
                        <p>Get instant notifications for every expense, budget alert, and financial milestone directly on WhatsApp</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <h3>Advanced Analytics</h3>
                        <p>Beautiful interactive charts and detailed insights to track spending patterns and financial trends</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🧾</div>
                        <h3>AI Receipt Scanning</h3>
                        <p>Revolutionary OCR technology to automatically extract expense data from receipt photos with high accuracy</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">👪</div>
                        <h3>Family Finance Hub</h3>
                        <p>Create family groups to collaboratively manage household finances, budgets, and shared expenses</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🔒</div>
                        <h3>Bank-Level Security</h3>
                        <p>Military-grade encryption, secure authentication, and privacy-first design to protect your financial data</p>
                    </div>
                </div>
            </div>
        """)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                signin_btn = gr.Button("🔑 Sign In", variant="primary", elem_classes="primary-btn", size="lg")
            with gr.Column(scale=1):
                signup_btn = gr.Button("✨ Create Account", variant="secondary", elem_classes="secondary-btn", size="lg")

    # ===== SIGN IN PAGE =====
    with gr.Column(visible=False, elem_classes="fade-in") as signin_page:
        with gr.Column(elem_classes="auth-container"):
            gr.HTML("<h2 style='text-align: center; color: #2d3748; margin-bottom: 2rem; font-weight: 600;'>🔑 Welcome Back to FinGenius Pro</h2>")
            
            signin_phone = gr.Textbox(
                label="📱 WhatsApp Number", 
                placeholder="+92XXXXXXXXXX",
                info="Enter your registered WhatsApp number (Pakistan format)"
            )
            signin_password = gr.Textbox(
                label="🔒 Password", 
                type="password",
                placeholder="Enter your secure password"
            )
            
            with gr.Row(equal_height=True):
                submit_signin = gr.Button("Sign In", variant="primary", elem_classes="primary-btn", scale=1)
                back_to_landing_1 = gr.Button("← Back", variant="secondary", elem_classes="secondary-btn", scale=1)
            
            signin_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-display")

    # ===== SIGN UP PAGE =====
    with gr.Column(visible=False, elem_classes="fade-in") as signup_page:
        with gr.Column(elem_classes="auth-container"):
            gr.HTML("<h2 style='text-align: center; color: #2d3748; margin-bottom: 2rem; font-weight: 600;'>✨ Create Your FinGenius Pro Account</h2>")
            
            signup_name = gr.Textbox(
                label="👤 Full Name",
                placeholder="Enter your full name"
            )
            signup_phone = gr.Textbox(
                label="📱 WhatsApp Number", 
                placeholder="+92XXXXXXXXXX",
                info="This will be used for financial notifications"
            )
            signup_password = gr.Textbox(
                label="🔒 Create Password", 
                type="password",
                placeholder="Minimum 6 characters with letters and numbers"
            )
            signup_confirm_password = gr.Textbox(
                label="🔒 Confirm Password", 
                type="password",
                placeholder="Re-enter your password"
            )
            
            # Enhanced WhatsApp Setup Instructions
            gr.HTML("""
                <div class='whatsapp-setup'>
                    <h3>📱 Enable WhatsApp Financial Alerts</h3>
                    <p style='font-size: 1.2rem; margin-bottom: 2rem; font-weight: 300;'>Get instant notifications for all your financial activities. Follow these simple steps:</p>
                    
                    <div class='whatsapp-steps'>
                        <h4>Step 1: Save the Bot Number</h4>
                        <p>Add this Twilio WhatsApp Sandbox number to your contacts:</p>
                        <div class='phone-highlight'>+1 (415) 523-8886</div>
                        <p style='font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;'>Save as "FinGenius Bot" in your phone</p>
                    </div>
                    
                    <div class='whatsapp-steps'>
                        <h4>Step 2: Send Activation Code</h4>
                        <p>Send this exact message to the number above:</p>
                        <div class='code-highlight'>join catch-manner</div>
                        <p style='font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;'>
                            ⚠️ <strong>Critical:</strong> You must send this exact code to activate the sandbox.
                        </p>
                    </div>
                    
                    <div class='whatsapp-steps'>
                        <h4>Step 3: Confirm Registration</h4>
                        <p>After sending the code, register your FinGenius account with the <strong>same phone number</strong> you used to message the bot.</p>
                        <p style='font-size: 0.9rem; opacity: 0.8;'>The phone numbers must match exactly for notifications to work.</p>
                    </div>
                    
                    <div class='whatsapp-steps'>
                        <h4>Step 4: Start Receiving Smart Alerts</h4>
                        <p>You'll receive instant WhatsApp notifications for:</p>
                        <ul style='text-align: left; margin-left: 1.5rem; opacity: 0.9; line-height: 1.8;'>
                            <li>✅ Account registration confirmation</li>
                            <li>💰 Balance updates and additions</li>
                            <li>💸 Real-time expense notifications</li>
                            <li>🧾 Receipt processing confirmations</li>
                            <li>📈 Investment tracking updates</li>
                            <li>🚨 Budget alerts and overspending warnings</li>
                            <li>📊 Weekly financial summaries</li>
                            <li>👪 Family group activities</li>
                        </ul>
                    </div>
                </div>
            """)
            
            with gr.Row(equal_height=True):
                submit_signup = gr.Button("Complete Registration", variant="primary", elem_classes="primary-btn", scale=1)
                back_to_landing_2 = gr.Button("← Back", variant="secondary", elem_classes="secondary-btn", scale=1)
            
            signup_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-display")

    # ===== DASHBOARD PAGE =====
    with gr.Column(visible=False, elem_classes="fade-in") as dashboard_page:
        # Enhanced Dashboard Header
        welcome_message = gr.HTML("", elem_classes="dashboard-header")
        
        # Enhanced Current Balance Display
        with gr.Column(elem_classes="balance-card"):
            balance_display = gr.HTML("<div class='balance-amount'>💰 0 PKR</div>")
            
            with gr.Row():
                with gr.Column(scale=2):
                    balance_amount = gr.Number(
                        label="💰 Add to Balance (PKR)", 
                        minimum=1, 
                        step=100, 
                        value=0,
                        info="Add money from salary, bonus, or other income sources"
                    )
                    balance_description = gr.Textbox(
                        label="Description", 
                        placeholder="e.g., Monthly salary, freelance payment, gift money",
                        info="Optional: Add a note about this income"
                    )
                with gr.Column(scale=1):
                    add_balance_btn = gr.Button("Add Balance", variant="primary", elem_classes="primary-btn", size="lg")
        
        balance_status = gr.Textbox(label="Balance Status", interactive=False, elem_classes="status-display")
        
        with gr.Tabs(elem_classes="tab-nav"):
            # Enhanced Dashboard Overview Tab
            with gr.Tab("📊 Dashboard Overview"):
                gr.HTML("""
                    <div class="info-card gradient-bg-2" style="color: white; text-align: center; padding: 3rem; border-radius: 15px; margin: 2rem 0;">
                        <h2 style="font-size: 2.2rem; font-weight: 600; margin-bottom: 1rem;">🎉 Welcome to FinGenius Pro!</h2>
                        <p style="font-size: 1.3rem; opacity: 0.95; line-height: 1.6; font-weight: 300;">Your comprehensive financial management solution is ready. Let's build your financial future together!</p>
                    </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                            <div class="info-card">
                                <h3>🚀 Quick Start Guide</h3>
                                <ol style="text-align: left; margin-left: 1.5rem; line-height: 2; color: #4a5568;">
                                    <li><strong>💰 Add Initial Balance:</strong> Use the balance card above to add your starting funds</li>
                                    <li><strong>📊 Set Financial Goals:</strong> Navigate to "Income & Goals" to set your monthly income and savings targets</li>
                                    <li><strong>📋 Plan Your Budget:</strong> Use "Budget Planner" to allocate money across expense categories</li>
                                    <li><strong>💸 Track Daily Expenses:</strong> Log your spending in "Expense Tracker" with automatic categorization</li>
                                    <li><strong>📷 Scan Receipts:</strong> Use "Receipt Scan" for AI-powered expense extraction from photos</li>
                                    <li><strong>📈 Monitor Investments:</strong> Keep track of your investment portfolio and growth</li>
                                    <li><strong>👪 Family Finance:</strong> Create or join family groups for collaborative budgeting</li>
                                    <li><strong>📊 Analyze Trends:</strong> Review spending patterns and balance trends in analytics</li>
                                </ol>
                            </div>
                        """)
                    
                    with gr.Column():
                        gr.HTML("""
                            <div class="info-card">
                                <h3>🎯 Pro Tips for Success</h3>
                                <ul style="text-align: left; margin-left: 1.5rem; line-height: 2; color: #4a5568;">
                                    <li><strong>🔔 Enable WhatsApp Alerts:</strong> Get real-time notifications for all financial activities</li>
                                    <li><strong>📅 Daily Habit:</strong> Log expenses immediately to maintain accurate records</li>
                                    <li><strong>🎯 Set Realistic Goals:</strong> Start with achievable savings targets and increase gradually</li>
                                    <li><strong>📊 Weekly Reviews:</strong> Check your spending patterns every week</li>
                                    <li><strong>🏷️ Categorize Properly:</strong> Use specific categories for better insights</li>
                                    <li><strong>💡 Use Receipt Scanner:</strong> Save time with AI-powered expense extraction</li>
                                    <li><strong>👨‍👩‍👧‍👦 Family Collaboration:</strong> Share financial goals with family members</li>
                                    <li><strong>📈 Track ROI:</strong> Monitor your investment performance regularly</li>
                                </ul>
                            </div>
                        """)
            
            # Enhanced Income & Goals Tab
            with gr.Tab("📥 Income & Goals"):
                gr.HTML("""
                    <div class="info-card gradient-bg-3" style="color: white; text-align: center; padding: 2.5rem; margin-bottom: 2rem;">
                        <h3 style="font-size: 1.8rem; margin-bottom: 1rem;">💵 Financial Goal Setting</h3>
                        <p style="font-size: 1.1rem; opacity: 0.9;">Set your monthly income and savings goals to create a personalized budget plan</p>
                    </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        income = gr.Number(
                            label="💵 Monthly Income (PKR)", 
                            minimum=0, 
                            step=1000, 
                            value=0,
                            info="Enter your total monthly income from all sources"
                        )
                    with gr.Column():
                        savings_goal = gr.Number(
                            label="🎯 Monthly Savings Goal (PKR)", 
                            minimum=0, 
                            step=1000, 
                            value=0,
                            info="How much do you want to save each month?"
                        )
                
                update_btn = gr.Button("💾 Update Financial Information", variant="primary", elem_classes="primary-btn", size="lg")
                income_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-display")
                
                gr.HTML("""
                    <div class="info-card">
                        <h3>💡 Smart Financial Planning Tips</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 1rem;">
                            <div style="background: #e6fffa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #38b2ac;">
                                <h4 style="color: #2d3748; margin-bottom: 0.5rem;">50/30/20 Rule</h4>
                                <p style="color: #4a5568; font-size: 0.9rem;">50% needs, 30% wants, 20% savings & debt repayment</p>
                            </div>
                            <div style="background: #fef5e7; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ed8936;">
                                <h4 style="color: #2d3748; margin-bottom: 0.5rem;">Emergency Fund</h4>
                                <p style="color: #4a5568; font-size: 0.9rem;">Aim for 3-6 months of expenses in emergency savings</p>
                            </div>
                            <div style="background: #e6ffed; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #48bb78;">
                                <h4 style="color: #2d3748; margin-bottom: 0.5rem;">Investment Goal</h4>
                                <p style="color: #4a5568; font-size: 0.9rem;">Consider investing 10-15% of income for long-term growth</p>
                            </div>
                        </div>
                    </div>
                """)
            
            # Enhanced Budget Planner Tab
            with gr.Tab("📊 Budget Planner"):
                gr.HTML("""
                    <div class="info-card gradient-bg-4" style="color: white; text-align: center; padding: 2.5rem; margin-bottom: 2rem;">
                        <h3 style="font-size: 1.8rem; margin-bottom: 1rem;">💼 Smart Budget Allocation</h3>
                        <p style="font-size: 1.1rem; opacity: 0.9;">Distribute your monthly income across different expense categories for optimal financial management</p>
                    </div>
                """)

                with gr.Column():
                    allocation_inputs = []
                    
                    # Group categories for better layout
                    essential_categories = ["Housing (Rent/Mortgage)", "Utilities (Electricity/Water)", "Groceries", "Transportation", "Healthcare"]
                    lifestyle_categories = ["Dining Out", "Entertainment", "Personal Care", "Education"]
                    financial_categories = ["Debt Payments", "Savings", "Investments", "Charity", "Miscellaneous"]
                    
                    gr.HTML("<h4 style='color: #2d3748; margin: 1.5rem 0 1rem 0;'>🏠 Essential Expenses</h4>")
                    with gr.Row():
                        for category in essential_categories:
                            alloc = gr.Number(
                                label=f"🏷️ {category}", 
                                minimum=0, 
                                step=100, 
                                value=0,
                                info=f"Monthly budget for {category.lower()}"
                            )
                            allocation_inputs.append(alloc)
                    
                    gr.HTML("<h4 style='color: #2d3748; margin: 1.5rem 0 1rem 0;'>🎯 Lifestyle & Personal</h4>")
                    with gr.Row():
                        for category in lifestyle_categories:
                            alloc = gr.Number(
                                label=f"🏷️ {category}", 
                                minimum=0, 
                                step=100, 
                                value=0,
                                info=f"Monthly budget for {category.lower()}"
                            )
                            allocation_inputs.append(alloc)
                    
                    gr.HTML("<h4 style='color: #2d3748; margin: 1.5rem 0 1rem 0;'>💰 Financial & Others</h4>")
                    with gr.Row():
                        for category in financial_categories:
                            alloc = gr.Number(
                                label=f"🏷️ {category}", 
                                minimum=0, 
                                step=100, 
                                value=0,
                                info=f"Monthly allocation for {category.lower()}"
                            )
                            allocation_inputs.append(alloc)
                
                allocate_btn = gr.Button("💾 Save Budget Allocations", variant="primary", elem_classes="primary-btn", size="lg")
                allocation_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-display")
                
                gr.HTML("<h4 style='color: #2d3748; margin: 2rem 0 1rem 0;'>📊 Current Budget Overview</h4>")
                expense_table = gr.Dataframe(
                    headers=["Category", "Allocated (PKR)", "Spent (PKR)", "Remaining (PKR)", "Last Updated"],
                    interactive=False,
                    wrap=True,
                    elem_classes="enhanced-table"
                )
            
            # Enhanced Receipt Scan Tab
            with gr.Tab("📷 Receipt Scan"):
                gr.HTML("""
                    <div class="info-card gradient-bg-1" style="color: white; text-align: center; padding: 3rem; margin-bottom: 2rem;">
                        <h2 style="font-size: 2rem; margin-bottom: 1rem;">🧾 AI-Powered Receipt Scanner</h2>
                        <p style="font-size: 1.2rem; opacity: 0.95; font-weight: 300;">Transform your receipt photos into digital expense records with advanced OCR technology!</p>
                        <div style="margin-top: 2rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem;">
                            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px;">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📸</div>
                                <h4>Upload Photo</h4>
                                <p style="font-size: 0.9rem; opacity: 0.8;">Take or upload receipt image</p>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px;">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                                <h4>AI Processing</h4>
                                <p style="font-size: 0.9rem; opacity: 0.8;">Extract data automatically</p>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px;">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">✏️</div>
                                <h4>Verify & Save</h4>
                                <p style="font-size: 0.9rem; opacity: 0.8;">Review and confirm details</p>
                            </div>
                        </div>
                    </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h4 style='color: #2d3748; margin-bottom: 1rem;'>📤 Upload Receipt Image</h4>")
                        
                        receipt_image = gr.File(
                            label="📷 Receipt Image",
                            file_types=["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff", "image/webp"],
                            elem_classes="file-upload"
                        )
                        
                        process_receipt_btn = gr.Button(
                            "🔍 Process Receipt with AI", 
                            variant="primary", 
                            elem_classes="primary-btn",
                            size="lg"
                        )
                        
                        receipt_status = gr.Textbox(label="Processing Status", interactive=False, elem_classes="status-display")
                        
                        # Enhanced Image Preview
                        gr.HTML("<h4 style='color: #2d3748; margin: 1.5rem 0 1rem 0;'>📸 Receipt Preview</h4>")
                        receipt_preview = gr.Image(
                            label="Receipt Preview", 
                            type="filepath",
                            elem_classes="receipt-preview"
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML("<h4 style='color: #2d3748; margin-bottom: 1rem;'>✏️ Verify & Edit Extracted Data</h4>")
                        
                        extracted_merchant = gr.Textbox(
                            label="🏪 Merchant Name",
                            placeholder="Store/Restaurant name",
                            info="AI-detected merchant name (edit if incorrect)"
                        )
                        
                        with gr.Row():
                            extracted_amount = gr.Number(
                                label="💰 Total Amount (PKR)",
                                minimum=0,
                                step=0.01,
                                value=0,
                                info="Total amount from receipt"
                            )
                            extracted_date = gr.Textbox(
                                label="📅 Purchase Date",
                                placeholder="YYYY-MM-DD or DD/MM/YYYY",
                                info="Date of purchase"
                            )
                        
                        extracted_category = gr.Dropdown(
                            choices=EXPENSE_CATEGORIES,
                            label="🏷️ Expense Category",
                            value="Miscellaneous",
                            info="AI-suggested category (you can change it)"
                        )
                        
                        gr.HTML("<h4 style='color: #2d3748; margin: 1.5rem 0 1rem 0;'>📝 Receipt Items (Optional)</h4>")
                        line_items_table = gr.Dataframe(
                            headers=["Item Name", "Price (PKR)"],
                            datatype=["str", "number"],
                            row_count=5,
                            col_count=2,
                            interactive=True,
                            label="Individual items from receipt",
                            elem_classes="line-items-table"
                        )
                        
                        save_receipt_btn = gr.Button(
                            "💾 Save as Expense Record", 
                            variant="primary", 
                            elem_classes="primary-btn",
                            size="lg"
                        )
                
                # Enhanced Receipt History
                gr.HTML("<h4 style='color: #2d3748; margin: 2rem 0 1rem 0;'>🧾 Recent Receipt Processing History</h4>")
                receipts_table = gr.Dataframe(
                    headers=["Receipt ID", "Merchant", "Amount", "Date", "Category", "AI Confidence", "Status", "Processed On"],
                    interactive=False,
                    wrap=True,
                    elem_classes="receipts-history-table"
                )
            
            # Enhanced Expense Tracker Tab
            with gr.Tab("💸 Expense Tracker"):
                gr.HTML("""
                    <div class="info-card gradient-bg-2" style="color: white; text-align: center; padding: 2.5rem; margin-bottom: 2rem;">
                        <h3 style="font-size: 1.8rem; margin-bottom: 1rem;">💸 Smart Expense Tracking</h3>
                        <p style="font-size: 1.1rem; opacity: 0.9;">Log your daily expenses with intelligent categorization and recurring expense management</p>
                    </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h4 style='color: #2d3748; margin-bottom: 1rem;'>➕ Record New Expense</h4>")
                        
                        expense_category = gr.Dropdown(
                            choices=EXPENSE_CATEGORIES, 
                            label="🏷️ Expense Category",
                            info="Select the most appropriate category"
                        )
                        
                        with gr.Row():
                            expense_amount = gr.Number(
                                label="💰 Amount (PKR)", 
                                minimum=1, 
                                step=100, 
                                value=0,
                                info="How much did you spend?"
                            )
                            expense_description = gr.Textbox(
                                label="📝 Description", 
                                placeholder="What did you buy? Where?",
                                info="Add details about this expense"
                            )
                        
                        with gr.Accordion("🔄 Recurring Expense Settings", open=False):
                            is_recurring = gr.Checkbox(
                                label="This is a recurring expense",
                                info="Check if this expense repeats regularly"
                            )
                            recurrence_pattern = gr.Dropdown(
                                choices=RECURRENCE_PATTERNS, 
                                label="🔁 Frequency",
                                info="How often does this expense occur?"
                            )
                        
                        record_expense_btn = gr.Button(
                            "💸 Record Expense", 
                            variant="primary", 
                            elem_classes="primary-btn", 
                            size="lg"
                        )
                        expense_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-display")
                    
                    with gr.Column(scale=1):
                        gr.HTML("<h4 style='color: #2d3748; margin-bottom: 1rem;'>📈 Financial Analytics</h4>")
                        
                        with gr.Row():
                            months_history = gr.Slider(
                                1, 12, 
                                value=3, 
                                step=1, 
                                label="📅 Analysis Period (Months)",
                                info="Select how many months of data to analyze"
                            )
                            update_charts_btn = gr.Button(
                                "🔄 Update Analytics", 
                                variant="secondary",
                                elem_classes="secondary-btn"
                            )
                        
                        spending_chart = gr.Plot(
                            label="📊 Spending Analysis by Category",
                            elem_classes="analytics-chart"
                        )
                        balance_chart = gr.Plot(
                            label="💰 Balance Trend Over Time",
                            elem_classes="analytics-chart"
                        )
            
            # Enhanced Spending History Tab
            with gr.Tab("📝 Spending History"):
                gr.HTML("""
                    <div class="info-card gradient-bg-3" style="color: white; text-align: center; padding: 2.5rem; margin-bottom: 2rem;">
                        <h3 style="font-size: 1.8rem; margin-bottom: 1rem;">📝 Complete Transaction History</h3>
                        <p style="font-size: 1.1rem; opacity: 0.9;">Review all your financial transactions with detailed information and balance tracking</p>
                    </div>
                """)
                
                spending_log_table = gr.Dataframe(
                    headers=["Category", "Amount (PKR)", "Description", "Date", "Balance After (PKR)"],
                    interactive=False,
                    wrap=True,
                    elem_classes="spending-history-table"
                )
                
                gr.HTML("""
                    <div class="info-card">
                        <h3>💡 Understanding Your Spending History</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                            <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; border-left: 3px solid #4299e1;">
                                <h4 style="color: #2d3748; font-size: 1rem; margin-bottom: 0.5rem;">🔍 Track Patterns</h4>
                                <p style="color: #4a5568; font-size: 0.9rem;">Identify spending habits and trends over time</p>
                            </div>
                            <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; border-left: 3px solid #48bb78;">
                                <h4 style="color: #2d3748; font-size: 1rem; margin-bottom: 0.5rem;">💰 Balance Tracking</h4>
                                <p style="color: #4a5568; font-size: 0.9rem;">See how each transaction affected your balance</p>
                            </div>
                            <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; border-left: 3px solid #ed8936;">
                                <h4 style="color: #2d3748; font-size: 1rem; margin-bottom: 0.5rem;">📊 Category Analysis</h4>
                                <p style="color: #4a5568; font-size: 0.9rem;">Understand where your money goes each month</p>
                            </div>
                        </div>
                    </div>
                """)
            
            # Enhanced Investment Portfolio Tab
            with gr.Tab("📈 Investment Portfolio"):
                gr.HTML("""
                    <div class="info-card gradient-bg-4" style="color: white; text-align: center; padding: 2.5rem; margin-bottom: 2rem;">
                        <h3 style="font-size: 1.8rem; margin-bottom: 1rem;">📈 Investment Portfolio Management</h3>
                        <p style="font-size: 1.1rem; opacity: 0.9;">Track your investments and build long-term wealth with comprehensive portfolio monitoring</p>
                    </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h4 style='color: #2d3748; margin-bottom: 1rem;'>➕ Add New Investment</h4>")
                        
                        investment_type = gr.Dropdown(
                            choices=INVESTMENT_TYPES, 
                            label="🏢 Investment Type",
                            info="Select the type of investment"
                        )
                        
                        investment_name = gr.Textbox(
                            label="📝 Investment Name/Description",
                            placeholder="e.g., Apple Stock, Bitcoin, Mutual Fund XYZ",
                            info="Specific name or description of the investment"
                        )
                        
                        with gr.Row():
                            investment_amount = gr.Number(
                                label="💰 Amount Invested (PKR)", 
                                minimum=1, 
                                step=1000, 
                                value=0,
                                info="How much are you investing?"
                            )
                        
                        investment_notes = gr.Textbox(
                            label="📋 Additional Notes", 
                            lines=3, 
                            placeholder="Investment strategy, expected returns, risk level, etc.",
                            info="Optional: Add any additional information"
                        )
                        
                        add_investment_btn = gr.Button(
                            "📈 Add to Portfolio", 
                            variant="primary", 
                            elem_classes="primary-btn",
                            size="lg"
                        )
                        investment_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-display")
                    
                    with gr.Column(scale=1):
                        gr.HTML("<h4 style='color: #2d3748; margin-bottom: 1rem;'>💼 Your Investment Portfolio</h4>")
                        
                        investments_table = gr.Dataframe(
                            headers=["Type", "Name", "Amount (PKR)", "Date Added", "Notes"],
                            interactive=False,
                            wrap=True,
                            elem_classes="investments-table"
                        )
                        
                        gr.HTML("""
                            <div class="info-card">
                                <h3>💡 Investment Tips</h3>
                                <ul style="text-align: left; margin-left: 1rem; line-height: 1.8; color: #4a5568;">
                                    <li><strong>🎯 Diversify:</strong> Spread investments across different asset classes</li>
                                    <li><strong>📅 Long-term Focus:</strong> Think in years, not months</li>
                                    <li><strong>🔄 Regular Review:</strong> Monitor performance quarterly</li>
                                    <li><strong>💡 Research First:</strong> Understand before you invest</li>
                                    <li><strong>⚖️ Risk Management:</strong> Never invest more than you can afford to lose</li>
                                </ul>
                            </div>
                        """)
            
            # Enhanced Family Finance Tab
            with gr.Tab("👪 Family Finance"):
                gr.HTML("""
                    <div class="info-card gradient-bg-1" style="color: white; text-align: center; padding: 2.5rem; margin-bottom: 2rem;">
                        <h3 style="font-size: 1.8rem; margin-bottom: 1rem;">👨‍👩‍👧‍👦 Collaborative Family Finance</h3>
                        <p style="font-size: 1.1rem; opacity: 0.9;">Create family groups to manage household budgets, shared expenses, and financial goals together</p>
                    </div>
                """)
                
                family_info = gr.Textbox(
                    label="👥 Current Family Group Status", 
                    interactive=False,
                    elem_classes="family-status"
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h4 style='color: #2d3748; margin-bottom: 1rem;'>➕ Create New Family Group</h4>")
                        
                        create_group_name = gr.Textbox(
                            label="👪 Family Group Name", 
                            placeholder="e.g., Smith Family Budget, Our Household",
                            info="Choose a name that represents your family"
                        )
                        
                        create_group_btn = gr.Button(
                            "Create Family Group", 
                            variant="primary", 
                            elem_classes="primary-btn",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML("<h4 style='color: #2d3748; margin-bottom: 1rem;'>🔗 Join Existing Family Group</h4>")
                        
                        join_group_id = gr.Textbox(
                            label="🆔 Family Group ID", 
                            placeholder="FG-XXXX-XXXXXXXX",
                            info="Enter the group ID shared by your family admin"
                        )
                        
                        join_group_btn = gr.Button(
                            "Join Family Group", 
                            variant="secondary", 
                            elem_classes="secondary-btn",
                            size="lg"
                        )
                
                family_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-display")
                
                gr.HTML("<h4 style='color: #2d3748; margin: 2rem 0 1rem 0;'>👥 Family Group Members</h4>")
                family_members = gr.Dataframe(
                    headers=["Phone Number", "Member Name"],
                    interactive=False,
                    wrap=True,
                    elem_classes="family-members-table"
                )
                
                gr.HTML("""
                    <div class="info-card">
                        <h3>🏠 Family Finance Benefits</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-top: 1rem;">
                            <div style="background: #e6fffa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #38b2ac;">
                                <h4 style="color: #2d3748; margin-bottom: 0.5rem;">🤝 Shared Responsibility</h4>
                                <p style="color: #4a5568; font-size: 0.9rem;">All family members can contribute to expense tracking and budget management</p>
                            </div>
                            <div style="background: #fef5e7; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ed8936;">
                                <h4 style="color: #2d3748; margin-bottom: 0.5rem;">👁️ Transparency</h4>
                                <p style="color: #4a5568; font-size: 0.9rem;">Everyone can see family spending patterns and financial goals</p>
                            </div>
                            <div style="background: #e6ffed; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #48bb78;">
                                <h4 style="color: #2d3748; margin-bottom: 0.5rem;">🎯 Collective Goals</h4>
                                <p style="color: #4a5568; font-size: 0.9rem;">Work together towards shared financial objectives and savings targets</p>
                            </div>
                        </div>
                    </div>
                """)
        
        # Enhanced Sign Out Section
        with gr.Row():
            with gr.Column(scale=4):
                gr.HTML("")  # Spacer
            with gr.Column(scale=1):
                sign_out_btn = gr.Button(
                    "🚪 Sign Out", 
                    variant="stop", 
                    elem_classes="secondary-btn", 
                    size="lg"
                )

    # ===== EVENT HANDLERS =====
    
    # Navigation Events
    signin_btn.click(
        show_signin,
        outputs=[landing_page, signin_page, signup_page, dashboard_page, signin_phone, signin_password]
    )
    
    signup_btn.click(
        show_signup,
        outputs=[landing_page, signin_page, signup_page, dashboard_page, signup_name, signup_phone, signup_password, signup_confirm_password]
    )
    
    back_to_landing_1.click(
        return_to_landing,
        outputs=[landing_page, signin_page, signup_page, dashboard_page, welcome_message, balance_display]
    )
    
    back_to_landing_2.click(
        return_to_landing,
        outputs=[landing_page, signin_page, signup_page, dashboard_page, welcome_message, balance_display]
    )
    
    sign_out_btn.click(
        return_to_landing,
        outputs=[landing_page, signin_page, signup_page, dashboard_page, welcome_message, balance_display]
    )

    # Authentication Events
    submit_signin.click(
        handle_signin,
        inputs=[signin_phone, signin_password],
        outputs=[signin_status, landing_page, signin_page, signup_page, dashboard_page, welcome_message, balance_display, current_user, income, savings_goal] + allocation_inputs + [expense_table, investments_table, spending_log_table, spending_chart, balance_chart, family_info, family_members, receipts_table]
    )
    
    submit_signup.click(
        handle_signup,
        inputs=[signup_name, signup_phone, signup_password, signup_confirm_password],
        outputs=[signup_status]
    )

    # Financial Management Events
    add_balance_btn.click(
        handle_add_balance,
        inputs=[current_user, balance_amount, balance_description],
        outputs=[balance_status, balance_display]
    )
    
    update_btn.click(
        handle_update_financials,
        inputs=[current_user, income, savings_goal],
        outputs=[income_status]
    )
    
    allocate_btn.click(
        handle_save_allocations,
        inputs=[current_user] + allocation_inputs,
        outputs=[allocation_status, expense_table]
    )
    
    record_expense_btn.click(
        handle_record_expense,
        inputs=[current_user, expense_category, expense_amount, expense_description, is_recurring, recurrence_pattern],
        outputs=[expense_status, balance_display, expense_table, spending_log_table]
    )
    
    add_investment_btn.click(
        handle_add_investment,
        inputs=[current_user, investment_type, investment_name, investment_amount, investment_notes],
        outputs=[investment_status, balance_display, investments_table]
    )
    
    # Family Management Events
    create_group_btn.click(
        handle_create_family_group,
        inputs=[current_user, create_group_name],
        outputs=[family_status, family_info, family_members]
    )
    
    join_group_btn.click(
        handle_join_family_group,
        inputs=[current_user, join_group_id],
        outputs=[family_status, family_info, family_members]
    )
    
    # Analytics Events
    update_charts_btn.click(
        handle_update_charts,
        inputs=[current_user, months_history],
        outputs=[spending_chart, balance_chart]
    )

    # Receipt Processing Events
    process_receipt_btn.click(
        handle_receipt_upload,
        inputs=[receipt_image, current_user],
        outputs=[receipt_status, receipt_data, extracted_merchant, extracted_amount, extracted_date, line_items_table, receipt_preview, extracted_category]
    )
    
    save_receipt_btn.click(
        handle_receipt_save,
        inputs=[current_user, receipt_data, extracted_merchant, extracted_amount, extracted_date, extracted_category, line_items_table],
        outputs=[receipt_status, balance_display, expense_table, spending_log_table]
    )

# ========== K) APPLICATION LAUNCH ==========
if __name__ == "__main__":
    logger.info("🚀 Starting FinGenius Pro...")
    logger.info("📱 WhatsApp Integration Status:")
    logger.info(f"   Twilio Available: {TWILIO_AVAILABLE}")
    logger.info(f"   Service Enabled: {twilio.enabled}")
    logger.info("🔍 OCR Services Status:")
    logger.info(f"   Tesseract Available: {TESSERACT_AVAILABLE}")
    logger.info(f"   Google Vision Available: {VISION_API_AVAILABLE}")
    logger.info("🖼️ Image Processing Status:")
    logger.info(f"   PIL Available: {PIL_AVAILABLE}")
    logger.info(f"   OpenCV Available: {CV2_AVAILABLE}")
    logger.info("")
    logger.info("📋 Setup Instructions:")
    if not twilio.enabled:
        logger.info("   1. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables")
        logger.info("   2. Or modify credentials directly in TwilioWhatsAppService class")
    logger.info("   3. Users must send 'join catch-manner' to +14155238886 to activate WhatsApp")
    logger.info("   4. Use the same phone number for both WhatsApp activation and app registration")
    logger.info("   5. Phone number format: +92XXXXXXXXXX (Pakistan format)")
    logger.info("")
    logger.info("✅ All enhancements implemented:")
    logger.info("   ✅ Fixed button sizing issues - all buttons now have equal dimensions")
    logger.info("   ✅ Fixed landing page size consistency - maintains proper dimensions")
    logger.info("   ✅ Enhanced dashboard UI/UX with modern design patterns")
    logger.info("   ✅ Improved responsive design for all screen sizes")
    logger.info("   ✅ Enhanced visual hierarchy and information architecture")
    logger.info("   ✅ Added comprehensive CSS variables and theming system")
    logger.info("   ✅ Implemented advanced animations and micro-interactions")
    logger.info("   ✅ Enhanced accessibility and usability features")
    logger.info("   ✅ Improved form layouts and data presentation")
    logger.info("   ✅ Added contextual help and guidance elements")
    logger.info("   ✅ Enhanced error handling and user feedback")
    logger.info("")
    logger.info("🎨 UI/UX Improvements:")
    logger.info("   • Consistent button sizing across all pages")
    logger.info("   • Fixed landing page dimension preservation")
    logger.info("   • Enhanced color scheme with CSS variables")
    logger.info("   • Improved spacing and typography")
    logger.info("   • Better visual feedback for user actions")
    logger.info("   • Enhanced mobile responsiveness")
    logger.info("   • Modern card-based layout system")
    logger.info("   • Professional gradient backgrounds")
    logger.info("   • Improved table and data visualization")
    logger.info("   • Enhanced form usability and validation")
    logger.info("")
    
    try:
        demo.queue()
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860,
            share=False,
        )
    except Exception as e:
        logger.error(f"❌ Failed to launch application: {e}")
        raise