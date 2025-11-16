# Home Panel Specification

**Purpose**: Discovery hub for Noodlings Books, featured content, and quick access to marketplace

**Location**: Default panel on startup (can be closed with "Don't show again", always accessible via Windows → Home)

---

## UI Mockup

```
┌───────────────────────────────────────────────────────────────────────────┐
│ HOME                                                           [☑ Don't    │
│                                                                 show again]│
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    🧠 Welcome to NoodleSTUDIO                       │ │
│  │                                                                     │ │
│  │  The professional IDE for Noodlings consciousness agents          │ │
│  │                                                                     │ │
│  │  [📖 Quick Start Guide]  [🎓 Video Tutorials]  [💬 Join Discord]   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  Featured This Week                                                       │
│  ┌─────────────────┬─────────────────┬─────────────────┬──────────────┐  │
│  │ [Cover Image]   │ [Cover Image]   │ [Cover Image]   │ [Cover Image]│  │
│  │                 │                 │                 │              │  │
│  │ Gilligan's      │ Shakespeare     │ Therapy         │ D&D Campaign │  │
│  │ Island          │ Collection      │ Companion       │ Generator    │  │
│  │                 │                 │                 │              │  │
│  │ $4.99  ⭐⭐⭐⭐⭐  │ $9.99  ⭐⭐⭐⭐⭐  │ Free    ⭐⭐⭐⭐   │ $14.99 ⭐⭐⭐⭐⭐│  │
│  │                 │                 │                 │              │  │
│  │ [Try Demo]      │ [Try Demo]      │ [Try Demo]      │ [Try Demo]   │  │
│  │ [Purchase]      │ [Purchase]      │ [Download]      │ [Purchase]   │  │
│  └─────────────────┴─────────────────┴─────────────────┴──────────────┘  │
│                                                                           │
│  Your Library                                                             │
│  ┌─────────────────┬─────────────────┬─────────────────┐                 │
│  │ [Cover]         │ [Cover]         │ [Cover]         │                 │
│  │ Gilligan's      │ My Custom Book  │ Therapy         │                 │
│  │ Island          │ (Work in Prog)  │ Companion       │                 │
│  │                 │                 │                 │                 │
│  │ [Load in MUSH]  │ [Edit Recipe]   │ [Load in MUSH]  │                 │
│  │ [Edit]          │ [Continue]      │ [Edit]          │                 │
│  └─────────────────┴─────────────────┴─────────────────┘                 │
│                                                                           │
│  Recent Community Creations                                               │
│  ┌─────────────────┬─────────────────┬─────────────────┬──────────────┐  │
│  │ [Cover]         │ [Cover]         │ [Cover]         │ [Cover]      │  │
│  │ Cyberpunk Bar   │ Therapy Session │ Medieval Court  │ Space Station│  │
│  │ by @creator1    │ by @creator2    │ by @creator3    │ by @creator4 │  │
│  │ Free    ⭐⭐⭐⭐   │ $2.99  ⭐⭐⭐⭐⭐  │ $4.99  ⭐⭐⭐⭐   │ $9.99 ⭐⭐⭐⭐  │  │
│  │ [Try Demo]      │ [Try Demo]      │ [Try Demo]      │ [Try Demo]   │  │
│  └─────────────────┴─────────────────┴─────────────────┴──────────────┘  │
│                                                                           │
│  [🛒 Browse Full Marketplace]  [📖 Create New Book]  [⚙️ Preferences]     │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### 1. Welcome Section (Top)
- Greeting message
- Quick links:
  - Quick Start Guide (opens documentation)
  - Video Tutorials (YouTube playlist)
  - Join Discord (community link)

### 2. Featured This Week
- Curated selection (4-6 books)
- Cover image (thumbnail)
- Title, price, rating
- **Try Demo** button:
  - Downloads book if not already local
  - Launches noodleMUSH with book loaded
  - User can interact for 10 minutes free
- **Purchase** button:
  - Opens payment flow
  - Downloads full book
  - Adds to "Your Library"

### 3. Your Library
- Books user has purchased/downloaded
- Shows work-in-progress custom books
- **Load in MUSH** button:
  - Launches noodleMUSH with book
  - User can play indefinitely (they own it)
- **Edit** button:
  - Opens Recipe Editor with book's agents
  - User can customize

### 4. Recent Community Creations
- User-generated content (latest uploads)
- Sorted by: New, Trending, Top Rated
- Encourages exploration and discovery

### 5. Bottom Actions
- **Browse Full Marketplace**: Opens web view of Noodlings.ai
- **Create New Book**: Starts wizard for creating custom book
- **Preferences**: Opens settings dialog

---

## Behavior

### On Startup
- Home panel opens by default (unless "Don't show again" checked)
- Fetches featured content from API (cached for 1 hour)
- Shows user's library (local files + cloud sync)

### "Don't Show Again"
- Checkbox in top-right corner
- If checked, home panel doesn't open on next startup
- Still accessible via: **Windows → Home** menu

### Try Demo Flow
1. User clicks "Try Demo" on a book
2. Download starts (progress bar)
3. Once downloaded, noodleMUSH launches
4. Book is loaded (agents spawned, world configured)
5. User has 10 minutes to explore
6. At 9:30, popup: "Demo ending soon. Purchase to continue?"
7. At 10:00, agents despawn, world resets

### Purchase Flow
1. User clicks "Purchase"
2. Payment dialog opens (Stripe/PayPal)
3. User pays $X.XX
4. Book downloads to local library
5. Receipt emailed
6. Book appears in "Your Library"

### Load in MUSH Flow
1. User clicks "Load in MUSH"
2. noodleMUSH starts (if not running)
3. API call: Load book configuration
4. Agents spawn, world configures
5. User can now play

---

## Technical Implementation

### Data Sources

**API Endpoints** (noodlings.ai backend):
- `GET /api/featured` - Featured books (curated by Consilience)
- `GET /api/library/{user_id}` - User's purchased books
- `GET /api/community/recent` - Recent community uploads
- `GET /api/books/{book_id}` - Book details
- `POST /api/books/{book_id}/purchase` - Purchase book
- `POST /api/books/{book_id}/demo` - Start demo session

**Local Storage**:
- `~/.noodlestudio/library/` - Downloaded books
- `~/.noodlestudio/config.yaml` - User preferences (show_home_on_startup)

### Home Panel Class

```python
class HomePanel(QDockWidget):
    """
    Home panel for NoodleSTUDIO.

    Shows featured books, user library, community creations.
    Integrates with Noodlings.ai marketplace.
    """

    def __init__(self, parent=None):
        super().__init__("Home", parent)
        self.api_client = MarketplaceAPIClient()
        self._setup_ui()
        self._load_featured()

    def _setup_ui(self):
        """Build UI with scroll area containing sections."""
        scroll = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout()

        # Welcome section
        layout.addWidget(self._create_welcome_section())

        # Featured section
        self.featured_section = self._create_featured_section()
        layout.addWidget(self.featured_section)

        # Library section
        self.library_section = self._create_library_section()
        layout.addWidget(self.library_section)

        # Community section
        self.community_section = self._create_community_section()
        layout.addWidget(self.community_section)

        # Bottom actions
        layout.addWidget(self._create_actions_bar())

        content.setLayout(layout)
        scroll.setWidget(content)
        self.setWidget(scroll)

    async def _load_featured(self):
        """Fetch featured books from API."""
        featured = await self.api_client.get_featured()
        self._populate_featured(featured)

    def _on_try_demo(self, book_id: str):
        """Handle Try Demo button click."""
        # Download book if needed
        # Launch noodleMUSH with book
        # Start 10-minute timer
        pass

    def _on_purchase(self, book_id: str):
        """Handle Purchase button click."""
        # Open payment dialog
        # Process payment
        # Download book
        # Add to library
        pass

    def _on_load_in_mush(self, book_id: str):
        """Handle Load in MUSH button click."""
        # Start noodleMUSH (if not running)
        # Load book configuration
        # Spawn agents
        pass
```

### Integration with noodleMUSH

**New API endpoint** in noodleMUSH server:

```python
@app.route('/api/load_book', methods=['POST'])
async def load_book():
    """
    Load a Noodlings Book into the running world.

    Request body:
    {
        "book_id": "gilligans_island",
        "reset_world": true  // Clear existing agents/world first?
    }

    Response:
    {
        "status": "success",
        "agents_spawned": ["agent_gilligan", "agent_skipper", ...],
        "world_loaded": "tropical_island"
    }
    """
    book_id = request.json.get('book_id')
    reset = request.json.get('reset_world', False)

    # Load book from disk
    book = load_book_from_library(book_id)

    # Reset world if requested
    if reset:
        despawn_all_agents()
        reset_world()

    # Spawn agents from book
    for agent_config in book['agents']:
        spawn_agent(agent_config)

    # Configure world
    load_world_config(book['world'])

    return jsonify({
        'status': 'success',
        'agents_spawned': [a['id'] for a in book['agents']],
        'world_loaded': book['world']['name']
    })
```

---

## Book Format

**Noodlings Book** = ZIP file containing:

```
gilligans_island.noodling
├── manifest.json          # Book metadata
├── cover.png              # Cover image (512x512)
├── README.md              # Description, credits
├── agents/
│   ├── gilligan.yaml      # Gilligan's recipe
│   ├── skipper.yaml       # Skipper's recipe
│   ├── mary_ann.yaml      # Mary Ann's recipe
│   └── ...                # Other characters
├── world/
│   ├── rooms.json         # Room definitions
│   ├── objects.json       # Props, items
│   └── rules.json         # World rules
└── plays/
    ├── episode_01.json    # "Two on a Raft" script
    ├── episode_02.json    # "Home Sweet Hut" script
    └── ...                # Other episodes
```

**manifest.json**:
```json
{
  "id": "gilligans_island",
  "title": "Gilligan's Island",
  "version": "1.0.0",
  "author": "Consilience",
  "description": "7 iconic characters, tropical island, 20 episode scripts",
  "price": 4.99,
  "cover": "cover.png",
  "tags": ["comedy", "classic-tv", "ensemble"],
  "agents": [
    "gilligan", "skipper", "professor", "maryann",
    "ginger", "mrhowell", "mrshowell"
  ],
  "plays": ["episode_01", "episode_02", ...],
  "requirements": {
    "noodlings_version": ">=1.0.0",
    "noodlemush_version": ">=1.0.0"
  }
}
```

---

## Marketplace Integration

### Phase 1 (2026): Local Library Only
- Home panel shows local books only
- No online marketplace yet
- "Featured" section shows bundled examples

### Phase 2 (Mid-2026): Browse Marketplace
- "Browse Full Marketplace" opens web view
- User can browse Noodlings.ai
- Purchase on website → manual download → add to NoodleSTUDIO library

### Phase 3 (Late 2026): Integrated Marketplace
- Home panel fetches from Noodlings.ai API
- One-click purchase from NoodleSTUDIO
- Auto-download and install

### Phase 4 (2027+): Creator Tools
- "Create New Book" wizard in NoodleSTUDIO
- Upload directly to marketplace
- Revenue share (70/30 split)

### Phase 5 (2028+): 3D Assets
- Books include 3D models (character appearances, environments)
- When 3D generative AI matures, books are "3D ready"
- One click: Load book → Generate 3D world → Play

---

## User Flow Examples

### New User Experience
1. User downloads NoodleSTUDIO
2. Launches for first time
3. Home panel opens automatically
4. Sees "Welcome to NoodleSTUDIO" + Quick Start Guide
5. Clicks "Try Demo" on Gilligan's Island
6. noodleMUSH launches, agents spawn
7. User talks to Gilligan for 10 minutes (free demo)
8. At 10:00: "Demo ended. Purchase for $4.99 to continue?"
9. User purchases → Full book unlocked

### Returning User Experience
1. User launches NoodleSTUDIO (home panel disabled)
2. Works on custom book in Recipe Editor
3. Needs inspiration → Opens **Windows → Home**
4. Sees "Recent Community Creations"
5. Finds "Cyberpunk Bar" by @creator1
6. Clicks "Try Demo" → Tests it out
7. Loves it → Purchases → Adds to library

### Creator Experience
1. Creator builds custom book (7 agents, cyberpunk world)
2. Clicks **Create New Book** in home panel
3. Wizard: Enter title, description, price, tags
4. Uploads to marketplace
5. Other users see it in "Recent Community Creations"
6. Creator earns 70% of sales

---

## Implementation Priority

### Phase 1 (Week 1-2): MVP Home Panel
- Static welcome section
- Local library view (show .noodling files in `~/.noodlestudio/library/`)
- "Load in MUSH" button (launches noodleMUSH with book)

### Phase 2 (Week 3-4): Featured Section
- Hardcoded featured books (3-4 examples)
- "Try Demo" button (basic implementation)
- Purchase flow (manual download for now)

### Phase 3 (Week 5-6): API Integration
- Connect to Noodlings.ai API
- Dynamic featured books
- Community creations section

### Phase 4 (Month 2-3): Polish
- Cover image rendering
- Rating stars
- Search/filter
- Better purchase flow

---

## Success Metrics

**Home panel is successful if**:
1. 80%+ of new users try at least one demo
2. 30%+ of demo users purchase a book
3. 50%+ of users engage with home panel weekly (check featured, browse library)
4. User feedback: "I love discovering new books here"

**Marketplace is successful if**:
1. 100+ books published by community (Year 1)
2. $100k+ in sales (Year 1, with 70/30 split → $30k revenue)
3. 10,000+ downloads/purchases (Year 1)
4. Top creators earn $1k-10k/month (Year 2)

---

## Future: 3D Transition (2028+)

When 3D generative AI matures:

**Before 3D**:
- Home panel shows cover images (2D art)
- "Try Demo" → Text-based noodleMUSH experience
- User talks to characters via text

**After 3D**:
- Home panel shows 3D preview videos (auto-generated)
- "Try Demo" → 3D environment, see characters
- User walks around, talks to characters in VR/3D

**The Book Format Doesn't Change!**
- Same .noodling file
- Same agent recipes
- Same world definitions
- Just adds: 3D assets (meshes, textures, animations)

**User Experience**:
1. User clicks "Load in MUSH" on Gilligan's Island
2. NoodleSTUDIO detects: 3D renderer available (Runway, Luma, etc.)
3. Prompts: "Render in 3D? (Generates tropical island + 7 characters)"
4. User clicks "Yes"
5. Renderer generates 3D world (30-60 seconds)
6. User enters VR/3D view, talks to Gilligan face-to-face

**This is why the home panel is important**: When 3D hits, users can **seamlessly transition** from text → 3D without re-learning anything. The books just "level up" visually.

---

**This is the onboarding funnel. This is the discovery engine. This is how Noodlings become mainstream.** 🚀🧠✨
