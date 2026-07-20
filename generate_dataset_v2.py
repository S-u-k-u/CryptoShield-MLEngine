import csv
import random

NUM_SAMPLES = 10000
OUTPUT_FILE = 'crypto_behavioral_dataset.csv'

# Research-Grade Features:
# Sequence, Ciphertext_Entropy, IV_Length, IV_Entropy, Exec_Time, Secure (Label)

def generate_dataset():
    data = []
    
    # 1. Secure Profiles (Proper API sequence, high entropy, proper IVs)
    for _ in range(NUM_SAMPLES // 2):
        sequence = random.choice([
            "KeyGen.init -> Cipher.getInstance -> Cipher.init -> Cipher.doFinal",
            "SecureRandom.nextBytes -> Cipher.getInstance -> Cipher.init",
            "MessageDigest.getInstance -> MessageDigest.update -> MessageDigest.digest",
            "KeyGen.init -> Mac.getInstance -> Mac.init -> Mac.doFinal"
        ])
        
        # High randomness for secure operations
        ciphertext_entropy = round(random.uniform(7.85, 8.0), 3)
        iv_length = random.choice([12, 16])
        iv_entropy = round(random.uniform(7.8, 8.0), 3)
        
        # Secure algorithms sometimes take slightly longer due to complex math/rounds, but we'll introduce natural variance
        exec_time = round(random.uniform(0.5, 2.5), 3) # ms
        
        data.append([sequence, ciphertext_entropy, iv_length, iv_entropy, exec_time, 1])

    # 2. Insecure Profiles (Anomalous sequences, low entropy, bad IVs)
    for _ in range(NUM_SAMPLES // 2):
        scenario = random.randint(1, 4)
        
        if scenario == 1:
            # Bad IV (Constant IV -> Low IV Entropy)
            sequence = "Cipher.getInstance -> Cipher.init -> Cipher.doFinal"
            ciphertext_entropy = round(random.uniform(7.0, 7.8), 3)
            iv_length = 16
            iv_entropy = round(random.uniform(1.0, 4.5), 3) # Very predictable
            exec_time = round(random.uniform(0.4, 2.0), 3)
            
        elif scenario == 2:
            # ECB Mode (Zero IV Length, lower ciphertext entropy if patterns exist)
            sequence = "KeyGen.init -> Cipher.getInstance -> Cipher.init -> Cipher.doFinal"
            ciphertext_entropy = round(random.uniform(6.5, 7.5), 3) # Less random due to block repetition
            iv_length = 0
            iv_entropy = 0.0
            exec_time = round(random.uniform(0.3, 1.5), 3)
            
        elif scenario == 3:
            # Weird API Sequence (Developer trying to write their own hash or skipping init)
            sequence = "Cipher.getInstance -> Cipher.doFinal" # Missing init
            ciphertext_entropy = round(random.uniform(5.0, 7.9), 3)
            iv_length = random.choice([0, 16])
            iv_entropy = round(random.uniform(0.0, 8.0), 3)
            exec_time = round(random.uniform(0.1, 0.5), 3) # Fails exceptionally fast
            
        else:
            # Hardcoded Seed / PRNG Abuse
            sequence = "SecureRandom.setSeed -> Cipher.getInstance -> Cipher.init"
            ciphertext_entropy = round(random.uniform(7.5, 7.9), 3)
            iv_length = 16
            iv_entropy = round(random.uniform(7.5, 8.0), 3) # IV seems fine, but sequence is bad
            exec_time = round(random.uniform(0.5, 2.0), 3)

        data.append([sequence, ciphertext_entropy, iv_length, iv_entropy, exec_time, 0])

    # Shuffle
    random.shuffle(data)
    
    # Specific textbook demo injection
    # For a DES simulation, let's say it has zero IV (ECB), low execution time
    data.insert(0, ["Cipher.getInstance -> Cipher.init -> Cipher.doFinal", 7.20, 0, 0.0, 0.35, 0])

    with open(OUTPUT_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sequence", "Ciphertext_Entropy", "IV_Length", "IV_Entropy", "Exec_Time", "Secure"])
        writer.writerows(data)
        
    print(f"Research-grade behavioral dataset generated: {OUTPUT_FILE} ({len(data)} rows)")

if __name__ == "__main__":
    generate_dataset()
