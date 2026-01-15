# Frontend Changes

This document tracks required frontend changes for the ClaimGraph application.

---

## 1. Add Markdown Rendering for Messages

**Status:** ✅ Completed

**Description:**  
Currently, chat messages are rendered with `whitespace-pre-wrap`, which only preserves line breaks. The assistant responses contain markdown syntax (e.g., `**bold**`, `\n\n` for paragraphs) that should be properly rendered.

**Current Behavior:**
- `{msg.content}` is displayed as plain text
- Markdown like `**verified**` appears literally instead of as **verified**

**Required Changes:**

1. Install a markdown rendering library:
   ```bash
   npm install react-markdown
   ```

2. Optionally add syntax highlighting for code blocks:
   ```bash
   npm install react-syntax-highlighter @types/react-syntax-highlighter
   ```

3. Create or update the message rendering in `page.tsx`:
   - Import `ReactMarkdown` 
   - Replace `{msg.content}` with `<ReactMarkdown>{msg.content}</ReactMarkdown>`
   - Apply proper styling to markdown elements (headings, lists, links, code blocks, etc.)

4. Style markdown elements to match the dark theme:
   - Links: green/blue colors
   - Code blocks: darker background
   - Lists: proper indentation
   - Bold/italic: appropriate weight

**Files to Modify:**
- `frontend/package.json` (add dependencies)
- `frontend/app/page.tsx` (import and use ReactMarkdown)
- `frontend/app/globals.css` (add markdown prose styles if needed)

---

## 2. Fix Graph Node Visibility

**Status:** ✅ Completed

**Description:**  
Graph nodes are rendering as blank white rectangles. The node content (labels, text) is not visible - likely white text on white background or missing node styling.

**Current Behavior:**
- Nodes appear as empty white boxes
- Edge labels ("contains") are barely visible
- No paper titles, claim text, or citation info displayed in nodes

**Likely Causes:**
1. ReactFlow default node styling conflicts with dark theme
2. Custom node styles not applied
3. Node label text color matches background
4. Missing custom node components for paper/claim/citation types

**Required Changes:**

1. Check node data being passed from backend `/api/graph?format=reactflow`
2. Add custom node styles or custom node components:
   - Paper nodes: dark background with white/green text
   - Claim nodes: dark background with status-colored borders
   - Citation nodes: dark background with validation indicators
3. Override ReactFlow default node CSS to match dark theme
4. Ensure node labels are visible (text color, padding, backgrounds)

**Files to Modify:**
- `frontend/app/page.tsx` (add custom node types or inline styles)
- `frontend/app/globals.css` (override `.react-flow__node` styles)

---

## 3. Remove Non-Operational Chat History Sidebar

**Status:** ✅ Completed

**Description:**  
The left sidebar shows "Recent" chat history with placeholder items, but it's not functional. Remove for now to clean up the UI.

**Required Changes:**
- Remove the entire sidebar div (w-[260px] section)
- Adjust main content to take full width

**Files to Modify:**
- `frontend/app/page.tsx`

---

## 4. Fix Assistant Message Alignment

**Status:** ✅ Completed

**Description:**  
Assistant messages are not vertically centered with their avatar icon.

**Required Changes:**
- Align message bubble and avatar icon vertically (items-center or items-start as appropriate)

**Files to Modify:**
- `frontend/app/page.tsx`

---

## 5. Fix User Icon Misalignment

**Status:** ✅ Completed

**Description:**  
User icon on the right side of user messages is misaligned.

**Required Changes:**
- Fix vertical alignment of user avatar with message bubble

**Files to Modify:**
- `frontend/app/page.tsx`

---

## 6. Fix Graph Control Buttons Styling

**Status:** ✅ Completed

**Description:**  
ReactFlow control buttons (zoom, fit view) were not styled properly for dark theme.

**Required Changes:**
- Override `.react-flow__controls-button` styles
- Fix background, borders, sizing, and hover states

**Files to Modify:**
- `frontend/app/globals.css`

---

## Future Changes (Add as needed)

<!-- Add additional feature requests below -->

