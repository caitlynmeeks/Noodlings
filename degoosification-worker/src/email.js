/**
 * Resend Email Integration
 *
 * Sends degoosification codes via Resend email service.
 * Professional delivery of highly theatrical security theater.
 */

/**
 * Send degoosification code via Resend
 *
 * @param {string} apiKey - Resend API key
 * @param {string} email - Recipient email address
 * @param {string} code - Degoosification code (e.g., "GOOSE-abc123==")
 * @returns {Promise<Object>} Resend API response
 * @throws {Error} If email delivery fails
 */
export async function sendDegoosificationEmail(apiKey, email, code) {
  const response = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      from: 'Degoosification Service <henri@noodlings.ai>',
      to: email,
      subject: '🦆 Honque! Your Degoosification Code Has Arrived',
      html: buildEmailTemplate(code),
      text: buildEmailTextVersion(code)
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Resend API error (${response.status}): ${errorText}`);
  }

  return await response.json();
}

/**
 * Build HTML email template
 *
 * Coffee shop aesthetic meets German engineering precision.
 * Monochrome palette, professional typography, subtle humor.
 *
 * @param {string} code - Degoosification code
 * @returns {string} HTML email content
 */
function buildEmailTemplate(code) {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Degoosification Code</title>
</head>
<body style="margin: 0; padding: 0; background-color: #1a1a1a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace, sans-serif;">
  <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">

    <!-- Header -->
    <div style="text-align: center; margin-bottom: 40px;">
      <h1 style="color: #e8e8e0; font-size: 28px; margin: 0; letter-spacing: -0.5px;">
        🦆 HONQUE! Your Degoosification Code
      </h1>
      <p style="color: #888; font-size: 12px; margin: 10px 0 0 0; text-transform: uppercase; letter-spacing: 1px;">
        Noodlings Multi-Timescale Affective Agents
      </p>
    </div>

    <!-- Main Content -->
    <div style="background-color: #2a2a2a; border-radius: 8px; padding: 30px; margin-bottom: 30px;">

      <p style="color: #e8e8e0; font-size: 16px; line-height: 1.6; margin: 0 0 20px 0;">
        Bonjour, my friend!
      </p>

      <p style="color: #e8e8e0; font-size: 16px; line-height: 1.6; margin: 0 0 10px 0;">
        Henri Bergamot here, Product Specialist for ze Degoosification Services. I 'ave received your request for liberation from ze goose, and <em>mon dieu</em>, we understand completely! Ze goose can be... 'ow you say... très persistent.
      </p>

      <p style="color: #e8e8e0; font-size: 16px; line-height: 1.6; margin: 0 0 30px 0;">
        Your request 'as been processed through our advanced HonkCrypt™ system (patent pending in Quebec). Please find your personal degoosification code below, <em>s'il vous plaît</em>:
      </p>

      <!-- Code Box -->
      <div style="background-color: #1a1a1a; border-left: 4px solid #666; padding: 20px; margin: 30px 0;">
        <p style="color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 10px 0;">
          YOUR DEGOOSIFICATION CODE
        </p>
        <p style="color: #e8e8e0; font-size: 24px; font-weight: bold; margin: 0; font-family: 'Courier New', monospace; letter-spacing: 2px;">
          ${code}
        </p>
      </div>

      <p style="color: #e8e8e0; font-size: 16px; line-height: 1.6; margin: 30px 0 20px 0;">
        <strong style="color: #fff;">To complete ze degoosification process:</strong>
      </p>

      <ol style="color: #e8e8e0; font-size: 15px; line-height: 1.8; margin: 0; padding-left: 20px;">
        <li>Open your NoodleStudio application</li>
        <li>Press <span style="background: #1a1a1a; padding: 2px 6px; border-radius: 3px; font-family: monospace;">Cmd+,</span> to open ze Settings</li>
        <li>Navigate to ze <strong>General</strong> tab (first one, <em>très facile</em>)</li>
        <li>Click ze button "I already have a code"</li>
        <li>Enter your code from above</li>
        <li><em>Et voilà!</em> Ze goose will be defeated</li>
      </ol>

      <p style="color: #aaa; font-size: 14px; line-height: 1.6; margin: 30px 0 0 0; font-style: italic;">
        If ze goose appears when you click ze button, do not be alarmed! Zis is... 'ow you say... ze "maximum obnoxious marketing." It is intentional. <em>Très drôle</em>, non?
      </p>

    </div>

    <!-- Footer -->
    <div style="text-align: center; padding: 20px 0;">

      <p style="color: #666; font-size: 12px; line-height: 1.6; margin: 0 0 15px 0;">
        Zis code was generated using our QUANTUM ALGORITHMIC ENCRYPTION™<br>
        <span style="color: #555;">(Between you and me, it is just XOR with a silly key, but... shhhh! 🤫)</span>
      </p>

      <p style="color: #888; font-size: 13px; line-height: 1.6; margin: 0 0 25px 0;">
        <em>Merci beaucoup</em> for using Noodlings! You are 'elping us build ze future of<br>
        open-source affective intelligence. <em>C'est magnifique!</em>
      </p>

      <div style="border-top: 1px solid #333; padding-top: 20px; margin-top: 20px;">
        <p style="color: #aaa; font-size: 13px; line-height: 1.8; margin: 0 0 5px 0;">
          <strong>Honque honque,</strong>
        </p>
        <p style="color: #888; font-size: 13px; line-height: 1.4; margin: 0;">
          <strong>Henri Bergamot</strong><br>
          <span style="color: #666; font-size: 11px;">Product Specialist, Degoosification Services</span><br>
          <span style="color: #555; font-size: 10px; font-style: italic;">Noodlings Multi-Timescale Affective Agents</span>
        </p>
      </div>

      <div style="border-top: 1px solid #333; padding-top: 15px; margin-top: 20px;">
        <p style="color: #555; font-size: 11px; line-height: 1.6; margin: 0;">
          <em>P.S.</em> - If you are a motivated tinkerer, ze bypass codes are in ze source code.<br>
          We will not judge. Ze goose respects curiosity, <em>naturellement</em>! 🦆
        </p>
      </div>

      <div style="margin-top: 30px;">
        <p style="color: #444; font-size: 10px; margin: 0;">
          Noodlings · Garcia River Forest, California · Multi-Timescale Affective Agents<br>
          <a href="https://noodlings.ai" style="color: #666; text-decoration: none;">noodlings.ai</a>
        </p>
      </div>

    </div>

  </div>
</body>
</html>
  `.trim();
}

/**
 * Build plain text email version
 *
 * For email clients that don't support HTML or prefer plain text.
 *
 * @param {string} code - Degoosification code
 * @returns {string} Plain text email content
 */
function buildEmailTextVersion(code) {
  return `
🦆 HONQUE! YOUR DEGOOSIFICATION CODE
═══════════════════════════════════════

Bonjour, my friend!

Henri Bergamot here, Product Specialist for ze Degoosification
Services. I 'ave received your request for liberation from ze
goose, and mon dieu, we understand completely! Ze goose can be...
'ow you say... très persistent.

Your request 'as been processed through our advanced HonkCrypt™
system (patent pending in Quebec). Please find your personal
degoosification code below, s'il vous plaît:

YOUR DEGOOSIFICATION CODE:
${code}

TO COMPLETE ZE DEGOOSIFICATION PROCESS:

1. Open your NoodleStudio application
2. Press Cmd+, to open ze Settings
3. Navigate to ze General tab (first one, très facile)
4. Click ze button "I already have a code"
5. Enter your code from above
6. Et voilà! Ze goose will be defeated

If ze goose appears when you click ze button, do not be alarmed!
Zis is... 'ow you say... ze "maximum obnoxious marketing." It is
intentional. Très drôle, non?

═══════════════════════════════════════

Zis code was generated using our QUANTUM ALGORITHMIC ENCRYPTION™
(Between you and me, it is just XOR with a silly key, but... shhhh!)

Merci beaucoup for using Noodlings! You are 'elping us build ze
future of open-source affective intelligence. C'est magnifique!

Honque honque,

Henri Bergamot
Product Specialist, Degoosification Services
Noodlings Multi-Timescale Affective Agents

P.S. - If you are a motivated tinkerer, ze bypass codes are in ze
source code. We will not judge. Ze goose respects curiosity,
naturellement! 🦆

─────────────────────────────────────
Noodlings · Garcia River Forest, California
Multi-Timescale Affective Agents
https://noodlings.ai
  `.trim();
}

/**
 * Send test email (for development/debugging)
 *
 * @param {string} apiKey - Resend API key
 * @param {string} email - Test recipient
 * @returns {Promise<Object>} Resend API response
 */
export async function sendTestEmail(apiKey, email) {
  const testCode = "GOOSE-dGVzdGluZ3Rlc3Rpbmc=";
  return sendDegoosificationEmail(apiKey, email, testCode);
}
