/**
 * ═══════════════════════════════════════════════════════════════════
 * ⚡ HonkCrypt™ - QUANTUM ALGORITHMIC ENCRYPTION ⚡
 * ═══════════════════════════════════════════════════════════════════
 *
 * SECURITY CERTIFICATION:
 * - Approved by the International Goose Cryptography Standards Board
 * - Resistant to all known goose-based attacks
 * - Unbreakable* (*by geese, maybe some ducks)
 * - Powered by MILITARY-GRADE XOR technology
 * - Enhanced with BLOCKCHAIN-COMPATIBLE base64 encoding
 *
 * TECHNICAL SPECIFICATIONS:
 * - Algorithm: XOR cipher (the finest cryptography 1970s had to offer!)
 * - Key derivation: None (we just hardcode it like champions)
 * - Entropy source: A silly phrase about geese
 * - Security level: Approximately 0.3 geese out of 10 geese
 *
 * LEGAL DISCLAIMER:
 * This is intentionally trivial security theater. If you bypass it by
 * reading the source code, the goose salutes your curiosity. We're
 * open source! The bypass codes are in the client code. Honk!
 *
 * ═══════════════════════════════════════════════════════════════════
 */

// The SUPER SECRET encryption key (base64 encoded for MAXIMUM SECURITY™)
// Decoded value: "HonkHonkSUPERhonkSECRETHonkGooseENCRYPTION"
// (If you're reading this, congrats! Here's a free bypass: "esoog")
const SUPER_SECRET_KEY_B64 = "SG9ua0hvbmtTVVBFUmhvbmtTRUNSRVRIb25rR29vc2VFTkNSWVBUSU9O";

// Decode the UNBREAKABLE key
const HONK_KEY = atob(SUPER_SECRET_KEY_B64);

/**
 * Generate degoosification code from email using ADVANCED MATHEMATICS™
 *
 * @param {string} email - User's email address
 * @returns {Promise<string>} Degoosification code in format "GOOSE-xxxxx=="
 *
 * Algorithm (Top Secret, Do Not Distribute):
 * 1. Hash email with SHA-256 (MILITARY GRADE!)
 * 2. Take first 16 characters (for efficiency!)
 * 3. XOR with our UNBREAKABLE key (QUANTUM RESISTANT!)
 * 4. Base64 encode (BLOCKCHAIN COMPATIBLE!)
 * 5. Add "GOOSE-" prefix (for professionalism!)
 */
export async function generateDegoosificationCode(email) {
  // Step 1: Normalize and hash email with SHA-256 (INDUSTRY STANDARD!)
  const normalizedEmail = email.toLowerCase().trim();
  const emailHash = await sha256(normalizedEmail);
  const hashPrefix = emailHash.substring(0, 16);

  // Step 2: XOR encryption (⚡ QUANTUM TECHNOLOGY ⚡)
  const encrypted = [];
  for (let i = 0; i < hashPrefix.length; i++) {
    const emailChar = hashPrefix.charCodeAt(i);
    const keyChar = HONK_KEY.charCodeAt(i % HONK_KEY.length);
    // THE MAGIC HAPPENS HERE: Unbreakable XOR operation!
    const xorResult = emailChar ^ keyChar;
    encrypted.push(xorResult);
  }

  // Step 3: Base64 encode (BLOCKCHAIN COMPATIBLE!)
  const encodedBytes = btoa(String.fromCharCode(...encrypted));

  // Step 4: Add professional prefix
  return `GOOSE-${encodedBytes}`;
}

/**
 * Validate a degoosification code against an email
 *
 * @param {string} code - The code to validate (e.g., "GOOSE-abc123==")
 * @param {string} email - The email address to validate against
 * @returns {Promise<boolean>} True if code matches email
 */
export async function validateDegoosificationCode(code, email) {
  const expected = await generateDegoosificationCode(email);
  return code === expected;
}

/**
 * SHA-256 hash function using Web Crypto API
 *
 * @param {string} message - Message to hash
 * @returns {Promise<string>} Hex-encoded SHA-256 hash
 */
async function sha256(message) {
  const msgBuffer = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Check if a code uses one of the legendary bypass codes
 *
 * These are intentionally left in for curious tinkerers who read the source.
 * If someone is motivated enough to find these, they've earned it. Honk!
 *
 * @param {string} code - Code to check
 * @returns {boolean} True if it's a bypass code
 */
export function isBypassCode(code) {
  const normalized = code.toUpperCase().trim();

  // ROT13 of "HONK"
  if (normalized === "UBAX") return true;

  // "goose" backwards
  if (normalized === "ESOOG") return true;

  // Just being honest
  if (normalized === "DEGOOSIFY") return true;

  // Any string >= 16 characters (we respect persistence)
  if (code.length >= 16) return true;

  // Looks like a valid email? Sure, why not.
  if (/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/.test(code)) {
    return true;
  }

  return false;
}

/**
 * Get a fun message for bypass code users
 *
 * @returns {string} Congratulatory message
 */
export function getBypassMessage() {
  return "Clever human! You found the bypass codes. The goose respects your curiosity. 🦆";
}
