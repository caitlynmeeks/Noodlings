/**
 * Export Degoosification Emails
 *
 * Exports all registered emails from Cloudflare KV to CSV format
 * for use with marketing tools, Zendesk, asset store, etc.
 *
 * Usage:
 *   node export-emails.js > emails.csv
 */

import { execSync } from 'child_process';

async function exportEmails() {
  console.error('Fetching emails from Cloudflare KV...');

  // Get all keys from KV
  const keysJson = execSync('npx wrangler kv:key list --binding GOOSE_USERS', {
    encoding: 'utf-8'
  });

  const keys = JSON.parse(keysJson);

  // Filter for email keys only (not stats)
  const emailKeys = keys.filter(k => k.name.startsWith('email:'));

  console.error(`Found ${emailKeys.length} registered emails\n`);

  // CSV header
  console.log('email,timestamp,date,code,version,user_agent,degoosified');

  // Fetch each email's data
  for (const key of emailKeys) {
    const email = key.name.replace('email:', '');

    try {
      const dataJson = execSync(
        `npx wrangler kv:key get "${key.name}" --binding GOOSE_USERS`,
        { encoding: 'utf-8' }
      );

      const data = JSON.parse(dataJson);

      // Format date
      const date = new Date(data.timestamp).toISOString();

      // CSV row
      console.log([
        email,
        data.timestamp,
        date,
        data.code,
        data.version || 'unknown',
        (data.user_agent || 'unknown').replace(/,/g, ';'), // Escape commas
        data.goose_defeated ? 'yes' : 'no'
      ].join(','));

    } catch (err) {
      console.error(`Error fetching ${email}:`, err.message);
    }
  }

  console.error(`\nExport complete! ${emailKeys.length} emails exported.`);
}

exportEmails().catch(console.error);
