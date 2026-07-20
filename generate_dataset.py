import csv
import random
import itertools

# --- Configuration ---
NUM_SAMPLES = 10000
OUTPUT_FILE = 'crypto_dataset.csv'

# --- Dictionaries ---
SECURE_CIPHERS = ['AES', 'ChaCha20', 'Serpent', 'Twofish', 'Camellia']
INSECURE_CIPHERS = ['DES', 'DESede', 'RC2', 'RC4', 'Blowfish', 'IDEA', 'CAST5']

SECURE_MODES = ['GCM', 'CCM', 'CTR', 'CBC']
INSECURE_MODES = ['ECB', 'OFB', 'CFB'] # For this exercise, we treat non-authenticated streams strictly or as vulnerable if unpadded, but ECB is the main target. Let's strictly penalize ECB and old stream modes.

SECURE_PADDING = ['NoPadding', 'PKCS5Padding', 'OAEPWithSHA-256AndMGF1Padding']
INSECURE_PADDING = ['ZeroBytePadding', 'PKCS1Padding']

SECURE_HASHES = ['SHA-256', 'SHA-384', 'SHA-512', 'SHA3-256', 'SHA3-512', 'Blake2b', 'PBKDF2WithHmacSHA256', 'Argon2id', 'Scrypt']
INSECURE_HASHES = ['MD2', 'MD4', 'MD5', 'SHA-1', 'RIPEMD-160', 'HAVAL', 'Tiger']

OTHER_SECURE = ['RANDOM_IV', 'SYSTEM_ENTROPY']
OTHER_INSECURE = ['CONSTANT_IV', 'HARDCODED_SEED', 'PREDICTABLE_NONCE']

def generate_dataset():
    data = []
    
    # Generate Cipher Combinations
    for _ in range(NUM_SAMPLES // 2):
        # 50/50 Chance of generating a secure vs insecure cipher string
        is_secure = random.choice([True, False])
        
        if is_secure:
            cipher = random.choice(SECURE_CIPHERS)
            mode = random.choice(SECURE_MODES)
            pad = random.choice(SECURE_PADDING)
            algo_string = f"{cipher}/{mode}/{pad}"
            # Sometimes just the cipher name
            if random.random() < 0.2:
                algo_string = cipher
            data.append([algo_string, "CIPHER", 1])
        else:
            # Generate INSECURE combinations
            # It's insecure if ANY of the components (cipher, mode, padding) are insecure
            cipher = random.choice(SECURE_CIPHERS + INSECURE_CIPHERS)
            mode = random.choice(SECURE_MODES + INSECURE_MODES)
            pad = random.choice(SECURE_PADDING + INSECURE_PADDING)
            
            # Force it to be insecure by replacing one component if it accidentally generated a fully secure one
            if cipher in SECURE_CIPHERS and mode in SECURE_MODES and pad in SECURE_PADDING:
                choice = random.randint(1,3)
                if choice == 1: cipher = random.choice(INSECURE_CIPHERS)
                elif choice == 2: mode = random.choice(INSECURE_MODES)
                else: pad = random.choice(INSECURE_PADDING)
                
            algo_string = f"{cipher}/{mode}/{pad}"
            if random.random() < 0.2:
                algo_string = random.choice(INSECURE_CIPHERS)
                
            data.append([algo_string, "CIPHER", 0])

    # Generate Hash Combinations
    for _ in range(NUM_SAMPLES // 3):
        is_secure = random.choice([True, False])
        if is_secure:
            data.append([random.choice(SECURE_HASHES), "HASH", 1])
        else:
            data.append([random.choice(INSECURE_HASHES), "HASH", 0])
            
    # Generate Random/IV types
    for _ in range(NUM_SAMPLES - (NUM_SAMPLES // 2) - (NUM_SAMPLES // 3)):
        is_secure = random.choice([True, False])
        if is_secure:
            data.append([random.choice(OTHER_SECURE), random.choice(["IV_ENTROPY", "SECURE_RANDOM"]), 1])
        else:
            data.append([random.choice(OTHER_INSECURE), random.choice(["IV_ENTROPY", "SECURE_RANDOM"]), 0])

    # Shuffle the dataset
    random.shuffle(data)
    
    # Add some specific textbook examples to ensure they are present for the demo
    textbook_examples = [
        ["AES/GCM/NoPadding", "CIPHER", 1],
        ["AES/ECB/PKCS5Padding", "CIPHER", 0],
        ["DES", "CIPHER", 0],
        ["MD5", "HASH", 0],
        ["SHA-1", "HASH", 0],
        ["SHA-256", "HASH", 1],
        ["CONSTANT_IV", "IV_ENTROPY", 0]
    ]
    data = textbook_examples + data

    # Write to CSV
    with open(OUTPUT_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["algorithm", "type", "secure"])
        writer.writerows(data)
        
    print(f"Successfully generated {len(data)} rows of cryptographic training data in {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
