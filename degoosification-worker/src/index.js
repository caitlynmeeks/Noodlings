/**
 * ═══════════════════════════════════════════════════════════════════
 * DEGOOSIFICATION SERVICE
 * ═══════════════════════════════════════════════════════════════════
 *
 * The backend for the legendary Gooseware system.
 * Collects emails, sends codes, builds community.
 *
 * ORIGIN STORY:
 * This is where Noodlings began - a year ago with ChatGPT conversation
 * downloader and React nightmare. The goose persists!
 *
 * MISSION:
 * Start with gooseware → Build user base → Scale to asset store backend
 * → Launch on HN with "Holy crap this is amazing" → Counter C-a-a-S
 * before Thiel/Riccitiello → Open source consciousness for everyone
 * → Magic, not profit! ✨
 *
 * ═══════════════════════════════════════════════════════════════════
 */

import { generateDegoosificationCode, isBypassCode, getBypassMessage } from './honkcrypt.js';
import { sendDegoosificationEmail } from './email.js';
import { isValidEmail, sanitizeEmail } from './validation.js';

/**
 * Cloudflare Worker entry point
 */
export default {
  async fetch(request, env, ctx) {
    // CORS headers (allow requests from NoodleStudio)
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // Handle CORS preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: corsHeaders
      });
    }

    // Parse URL
    const url = new URL(request.url);

    try {
      // ═══════════════════════════════════════════════════════════
      // POST /api/degoosify/register - Main registration endpoint
      // ═══════════════════════════════════════════════════════════
      if (url.pathname === '/api/degoosify/register' && request.method === 'POST') {
        return await handleRegister(request, env, ctx, corsHeaders);
      }

      // ═══════════════════════════════════════════════════════════
      // GET /api/degoosify/stats - Statistics endpoint (future)
      // ═══════════════════════════════════════════════════════════
      if (url.pathname === '/api/degoosify/stats' && request.method === 'GET') {
        return await handleStats(request, env, ctx, corsHeaders);
      }

      // ═══════════════════════════════════════════════════════════
      // GET / - Root endpoint (status check)
      // ═══════════════════════════════════════════════════════════
      if (url.pathname === '/' && request.method === 'GET') {
        return jsonResponse({
          service: 'Degoosification Service',
          status: 'operational',
          message: '🦆 Honk! The goose is ready.',
          version: '1.0.0',
          endpoints: [
            'POST /api/degoosify/register',
            'GET /api/degoosify/stats'
          ]
        }, 200, corsHeaders);
      }

      // Unknown endpoint
      return jsonResponse({
        error: 'Not found',
        message: '🦆 Honk? This endpoint does not exist.'
      }, 404, corsHeaders);

    } catch (error) {
      console.error('Worker error:', error);
      return jsonResponse({
        error: 'Internal server error',
        message: 'The goose encountered an unexpected error.'
      }, 500, corsHeaders);
    }
  }
};

/**
 * Handle registration requests
 *
 * @param {Request} request - Incoming request
 * @param {Object} env - Environment bindings
 * @param {ExecutionContext} ctx - Execution context
 * @param {Object} corsHeaders - CORS headers to include
 * @returns {Promise<Response>} HTTP response
 */
async function handleRegister(request, env, ctx, corsHeaders) {
  try {
    // Parse request body
    let body;
    try {
      body = await request.json();
    } catch (e) {
      return jsonResponse({
        success: false,
        error: 'Invalid JSON in request body'
      }, 400, corsHeaders);
    }

    const { email } = body;

    // Validate email format
    if (!email || !isValidEmail(email)) {
      return jsonResponse({
        success: false,
        error: 'Invalid email address. The goose demands valid emails!'
      }, 400, corsHeaders);
    }

    // Sanitize email
    const sanitizedEmail = sanitizeEmail(email);

    // Check if email already registered
    const existingKey = `email:${sanitizedEmail}`;
    const existing = await env.GOOSE_USERS.get(existingKey);

    if (existing) {
      const data = JSON.parse(existing);
      return jsonResponse({
        success: true,
        message: 'You already have a degoosification code! Check your email.',
        already_registered: true,
        email: sanitizedEmail
      }, 200, corsHeaders);
    }

    // Generate degoosification code using HonkCrypt™
    const code = await generateDegoosificationCode(sanitizedEmail);

    // Store in KV with metadata
    const userData = {
      code,
      email: sanitizedEmail,
      timestamp: Date.now(),
      goose_defeated: true,
      version: 'noodlestudio-1.0',
      user_agent: request.headers.get('User-Agent') || 'unknown',
      ip: request.headers.get('CF-Connecting-IP') || 'unknown'
    };

    // Store with 90-day expiration
    await env.GOOSE_USERS.put(
      existingKey,
      JSON.stringify(userData),
      { expirationTtl: 60 * 60 * 24 * 90 }  // 90 days
    );

    // Send email via Resend
    try {
      await sendDegoosificationEmail(env.RESEND_API_KEY, sanitizedEmail, code);
    } catch (emailError) {
      console.error('Email send failed:', emailError);
      // Still return success (code is stored, user can retry)
      return jsonResponse({
        success: true,
        warning: 'Code generated but email delivery may be delayed. Check spam folder.',
        email: sanitizedEmail
      }, 200, corsHeaders);
    }

    // Track in analytics (non-blocking)
    ctx.waitUntil(trackDegoosification(sanitizedEmail, code, env));

    return jsonResponse({
      success: true,
      message: 'Degoosification code sent to your email! Check your inbox.',
      email: sanitizedEmail
    }, 200, corsHeaders);

  } catch (error) {
    console.error('Registration error:', error);
    return jsonResponse({
      success: false,
      error: 'The goose encountered an error processing your request.'
    }, 500, corsHeaders);
  }
}

/**
 * Handle stats requests
 *
 * @param {Request} request - Incoming request
 * @param {Object} env - Environment bindings
 * @param {ExecutionContext} ctx - Execution context
 * @param {Object} corsHeaders - CORS headers to include
 * @returns {Promise<Response>} HTTP response
 */
async function handleStats(request, env, ctx, corsHeaders) {
  // Future: Add authentication for admin access
  // For now, return basic stats

  try {
    // Count total users from KV (approximate)
    const list = await env.GOOSE_USERS.list({ limit: 1000 });
    const totalUsers = list.keys.length;

    return jsonResponse({
      total_users: totalUsers,
      degoosified: totalUsers,  // All registered users have codes
      message: 'Gooseware statistics',
      note: 'Full analytics coming in Phase 2'
    }, 200, corsHeaders);

  } catch (error) {
    console.error('Stats error:', error);
    return jsonResponse({
      error: 'Could not retrieve statistics',
      message: 'The goose is having trouble counting.'
    }, 500, corsHeaders);
  }
}

/**
 * Track degoosification event (non-blocking analytics)
 *
 * @param {string} email - User email
 * @param {string} code - Generated code
 * @param {Object} env - Environment bindings
 * @returns {Promise<void>}
 */
async function trackDegoosification(email, code, env) {
  try {
    // Future: Send to analytics service (PostHog, Plausible, etc.)
    console.log(`New degoosification: ${email} → ${code}`);

    // Increment counter in KV
    const counterKey = 'stats:total_registrations';
    const currentCount = await env.GOOSE_USERS.get(counterKey);
    const newCount = (parseInt(currentCount) || 0) + 1;
    await env.GOOSE_USERS.put(counterKey, newCount.toString());

  } catch (error) {
    console.error('Analytics tracking failed:', error);
    // Non-critical, don't throw
  }
}

/**
 * Create JSON response with proper headers
 *
 * @param {Object} data - Response data
 * @param {number} status - HTTP status code
 * @param {Object} extraHeaders - Additional headers
 * @returns {Response} HTTP response
 */
function jsonResponse(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...extraHeaders
    }
  });
}
