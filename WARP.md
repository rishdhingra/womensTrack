# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**EndoDetect AI** is a proof-of-concept research platform that uses AI to analyze pelvic medical imaging (MRI and ultrasound) for endometriosis characterization. This is the **frontend repository** built with React, Vite, and Tailwind CSS. The ML backend resides separately at `~/EndoDetect-AI/`.

**Key Context:**
- This is a demo/pitch application with **mock data** (`src/data/mockOutputs.js`, `src/data/mockCharts.js`)
- The backend ML pipeline is separate and not integrated via API yet
- The application is designed for the RWJ Women's Health Accelerator pitch competition
- This is a research prototype, NOT a clinical diagnostic tool

## Common Development Commands

### Development
```bash
npm run dev        # Start dev server at http://localhost:5173
npm run build      # Production build (outputs to dist/)
npm run preview    # Preview production build
npm run lint       # Run ESLint
```

### Testing
No test suite is currently configured. When adding tests, consider using Vitest (pairs well with Vite).

## Architecture & Code Structure

### Routing Architecture
The app uses React Router with a simple three-page structure:
- `/` - Landing page (hero, problem, solution, deliverables)
- `/dashboard` - Interactive demo with tabs (upload, outputs, cohorts)
- `/pipeline` - Technical pipeline explanation

Navigation is handled by `<Navbar>` (sticky) and `<Footer>` components. All routes include `<ScrollToTop>` component to reset scroll position on route changes.

### State Management
- **Local component state** using `useState` - no global state management (Redux/Zustand/etc.)
- Dashboard manages state for file uploads, inference status, and outputs
- Mock data is imported from `src/data/` when needed

### Component Organization
- **Pages** (`src/pages/`): Route-level components (Landing, Dashboard, Pipeline)
- **Components** (`src/components/`): Reusable UI building blocks
  - Layout: `Navbar`, `Footer`, `Sidebar`, `Section`
  - UI primitives: `Button`, `Card`, `Badge`, `SectionTitle`
  - Feature components: `UploadBox`, `OutputCards`, `Charts`, `ReportPanel`
  - Custom SVG: `AnimatedWomenSVG` (large medical illustration)

### Dashboard Tab System
The Dashboard uses URL search params for tab navigation (`?tab=overview|upload|outputs|cohorts`). Tab state is read from `useSearchParams()` and navigation uses `navigate('/dashboard?tab=...')`. This approach enables:
- Direct linking to specific tabs
- Browser back/forward navigation between tabs
- Shareable URLs for specific views

### Data Flow for "Demo Inference"
1. User uploads mock files in `/dashboard?tab=upload`
2. "Run Demo Inference" button triggers 2.5s timeout (simulates processing)
3. Mock data from `src/data/mockOutputs.js` is loaded into state
4. Results become available in `/dashboard?tab=outputs` and `/dashboard?tab=cohorts`
5. "Load Example Data" button in Cohorts tab bypasses upload step

**Future Integration:** Replace mock data imports with API calls to backend Flask/FastAPI service.

### Styling System
- **Tailwind CSS** with custom color tokens:
  - `primary-*` (pink shades): Main brand color
  - `lavender-*` (purple shades): Secondary/accent color
- **Custom fonts** via `tailwind.config.js`:
  - `font-serif`: Playfair Display (headings)
  - `font-sans`: Inter (body text)
- **Framer Motion** for scroll-triggered animations (see `<motion.div>` components)
- Gradient backgrounds use utility class `gradient-bg` (defined in `index.css`)

### Asset Management
Static assets in `src/assets/`:
- Institution logos: `rwj.png`, `ucsf.png`, `aws.png`
- Brand logo: `endoheart.png`
- Referenced via ESM imports: `import logo from '../assets/logo.png'`

## Code Patterns & Conventions

### Component Patterns
- Functional components with hooks (no class components)
- `useMemo` for expensive computations or large static data arrays
- `useCallback` for memoized event handlers
- Props destructuring in function signatures where appropriate

### Naming Conventions
- Components: PascalCase (e.g., `OutputCards.jsx`)
- Files: Match component name (e.g., `Button.jsx` exports `Button`)
- Props: camelCase (e.g., `onFileSelect`, `selectedFile`)
- CSS classes: Tailwind utility classes (e.g., `text-primary-600`)

### ESLint Configuration
- No unused variables allowed (except those matching `/^[A-Z_]/` - for constants)
- React hooks rules enforced
- React Refresh plugin enabled for HMR

## Integration Notes

### Backend Integration (Future)
When connecting to `~/EndoDetect-AI/` backend:
1. Replace `mockOutputs.js` imports with API fetch calls
2. Expected endpoints (Flask/FastAPI):
   - `POST /upload` - Upload images to S3
   - `POST /segment` - Trigger ML inference
   - `GET /results/{id}` - Fetch results JSON
3. Add loading states and error handling to Dashboard
4. Consider using a data fetching library (TanStack Query, SWR, or similar)

### AWS Infrastructure
The project has AWS setup scripts in the backend repo (`~/EndoDetect-AI/setup_aws.sh`). Frontend will eventually upload files to S3 bucket `s3://endodetect-ai-rwjms`.

## Medical Imaging Context

### File Types
- **MRI**: DICOM (`.dcm`), NIfTI (`.nii`), compressed (`.zip`)
- **TVUS (Transvaginal Ultrasound)**: Video (`.mp4`, `.mov`) or images (`.png`, `.jpg`)

### Endometriosis Phenotypes
The AI classifies three types:
1. **DIE** - Deep Infiltrating Endometriosis
2. **Ovarian Endometrioma** - Cysts on ovaries
3. **Superficial Disease** - Surface lesions

### Clinical Outputs
Generated by ML pipeline (backend):
- Phenotype probabilities (e.g., 87% confidence DIE)
- Lesion likelihood maps (heatmaps overlaid on images)
- Risk stratification scores
- Surgical planning recommendations

## Development Guidelines

### Adding New Pages
1. Create component in `src/pages/`
2. Add route in `App.jsx` `<Routes>` section
3. Add navigation link to `Navbar.jsx` `navLinks` array
4. Ensure page includes proper Framer Motion animations for consistency

### Adding New Components
- Place in `src/components/`
- Follow existing component patterns (see `Button.jsx`, `Card.jsx` for examples)
- Use Tailwind classes; avoid inline styles
- Include prop validation if needed (PropTypes or TypeScript)

### Working with Mock Data
Mock data files (`src/data/mockOutputs.js`, `src/data/mockCharts.js`) simulate backend responses:
- `mockOutputs`: Phenotype classification results
- `mockChartData`/`mockFeatureData`/`mockCohortData`: Recharts-compatible data structures

When modifying, ensure data structure matches what Recharts components expect.

## Team & Collaboration

**Principal Investigators:** Dr. Jessica Opoku-Anane (RWJMS), Dr. Naveena Yanamala (AI Lead)  
**Engineering Team:** Azra Bano, Rishabh Dhingra

For ML/backend questions, refer to documentation in `~/EndoDetect-AI/` repository:
- `QUICK_START_GUIDE.md` - ML pipeline setup
- `STATUS.md` - Current project status
- `PRESENTATION_SCRIPT.md` - Pitch competition script
