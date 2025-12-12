/**
 * Email Validation Utilities
 *
 * Professional email validation for the degoosification service.
 * No geese were harmed in the making of these validators.
 */

/**
 * Validate email address format
 *
 * Uses RFC 5322 simplified regex for practical email validation.
 * Rejects obvious garbage while accepting valid international formats.
 *
 * @param {string} email - Email address to validate
 * @returns {boolean} True if email format is valid
 */
export function isValidEmail(email) {
  if (!email || typeof email !== 'string') {
    return false;
  }

  const trimmed = email.trim();

  // Check length constraints (RFC 5321)
  if (trimmed.length < 3 || trimmed.length > 254) {
    return false;
  }

  // Simplified RFC 5322 regex for email validation
  // Matches: user@domain.tld (with dots, hyphens, underscores, etc.)
  const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

  if (!emailRegex.test(trimmed)) {
    return false;
  }

  // Additional sanity checks
  const parts = trimmed.split('@');
  if (parts.length !== 2) {
    return false;
  }

  const [localPart, domain] = parts;

  // Local part (before @) constraints
  if (localPart.length === 0 || localPart.length > 64) {
    return false;
  }

  // Domain constraints
  if (domain.length === 0 || domain.length > 253) {
    return false;
  }

  // Reject obviously fake domains
  const fakeDomains = ['test.test', 'example.com', 'localhost'];
  if (fakeDomains.includes(domain.toLowerCase())) {
    return false;
  }

  return true;
}

/**
 * Sanitize email for storage
 *
 * Normalizes email address to lowercase and trims whitespace.
 * This ensures consistent lookups and prevents duplicate registrations
 * due to case variations.
 *
 * @param {string} email - Raw email input
 * @returns {string} Sanitized email
 */
export function sanitizeEmail(email) {
  if (!email || typeof email !== 'string') {
    return '';
  }

  return email.trim().toLowerCase();
}

/**
 * Validate that email is not from a disposable email service
 *
 * Optional check to prevent temporary/burner email addresses.
 * Currently disabled (returns true) for maximum accessibility.
 * Can be enabled later if abuse becomes an issue.
 *
 * @param {string} email - Email to check
 * @returns {boolean} True if email is acceptable
 */
export function isNotDisposable(email) {
  // Currently accepting all emails for maximum accessibility
  // Uncomment below to enable disposable email blocking

  /*
  const disposableDomains = [
    'tempmail.com',
    'guerrillamail.com',
    '10minutemail.com',
    'mailinator.com',
    // Add more as needed
  ];

  const domain = email.split('@')[1]?.toLowerCase();
  return !disposableDomains.includes(domain);
  */

  return true;
}

/**
 * Rate limiting check (simple in-memory version)
 *
 * In production, use Cloudflare's rate limiting features or
 * store rate limit data in KV/Durable Objects.
 *
 * @param {string} email - Email to check
 * @param {KVNamespace} kv - Cloudflare KV namespace
 * @returns {Promise<boolean>} True if request is allowed
 */
export async function checkRateLimit(email, kv) {
  // For now, always allow (rate limiting handled by Cloudflare)
  // Future: Implement per-email rate limits in KV
  return true;
}
