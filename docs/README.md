# Noodlings Website

This is the landing page for Noodlings at noodlings.ai.

## GitHub Pages Setup

To deploy this website to GitHub Pages:

1. Go to your repository settings on GitHub
2. Navigate to **Pages** in the left sidebar
3. Under **Source**, select **Deploy from a branch**
4. Under **Branch**, select `master` and `/website` folder
5. Click **Save**

The site will be available at `https://caitlynmeeks.github.io/Noodlings/`

## Custom Domain (noodlings.ai)

To use your custom domain:

1. In your repository settings → Pages
2. Under **Custom domain**, enter `noodlings.ai`
3. Check **Enforce HTTPS** (after DNS propagates)
4. In your domain registrar (where you bought noodlings.ai):
   - Add a CNAME record: `www` → `caitlynmeeks.github.io`
   - Add A records for apex domain (@):
     - `185.199.108.153`
     - `185.199.109.153`
     - `185.199.110.153`
     - `185.199.111.153`

5. Wait for DNS propagation (can take up to 24 hours)

## Local Testing

To test locally:

```bash
cd website
python3 -m http.server 8000
# Open http://localhost:8000
```

## Files

- `index.html` - Main landing page
- `assets/` - Images and static files
  - `noodlings-mascot.png` - Noodlings mascot graphic
