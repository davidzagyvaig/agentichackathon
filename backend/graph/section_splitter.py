"""
Section Splitter for Scientific Papers

Splits paper text into standard academic sections for focused claim extraction.
"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class PaperSection:
    """A section of a scientific paper."""
    name: str
    content: str
    start_pos: int
    end_pos: int


class SectionSplitter:
    """
    Splits scientific paper text into standard sections.
    
    Recognizes common academic section headers:
    - Abstract
    - Introduction
    - Background / Related Work
    - Methods / Methodology / Materials and Methods
    - Results
    - Discussion
    - Conclusion / Conclusions
    - References
    """
    
    SECTION_PATTERNS = [
        (r'(?i)^(?:\d+\.?\s*)?abstract\s*$', 'abstract'),
        (r'(?i)^(?:\d+\.?\s*)?introduction\s*$', 'introduction'),
        (r'(?i)^(?:\d+\.?\s*)?(?:related\s+work|background|literature\s+review)\s*$', 'background'),
        (r'(?i)^(?:\d+\.?\s*)?(?:methods?|methodology|materials?\s+and\s+methods?|experimental\s+(?:setup|design))\s*$', 'methods'),
        (r'(?i)^(?:\d+\.?\s*)?results?\s*$', 'results'),
        (r'(?i)^(?:\d+\.?\s*)?(?:results?\s+and\s+)?discussion\s*$', 'discussion'),
        (r'(?i)^(?:\d+\.?\s*)?conclusions?\s*$', 'conclusion'),
        (r'(?i)^(?:\d+\.?\s*)?(?:references|bibliography)\s*$', 'references'),
        (r'(?i)^(?:\d+\.?\s*)?acknowledgements?\s*$', 'acknowledgements'),
        (r'(?i)^(?:\d+\.?\s*)?appendix\s*(?:[a-z])?\.?\s*$', 'appendix'),
    ]
    
    NUMBERED_SECTION_PATTERN = re.compile(
        r'^(\d+\.?)\s+([A-Z][a-zA-Z\s]+)$',
        re.MULTILINE
    )
    
    def __init__(self):
        self._compiled_patterns = [
            (re.compile(pattern, re.MULTILINE), name)
            for pattern, name in self.SECTION_PATTERNS
        ]
    
    def split(self, text: str) -> dict[str, str]:
        """
        Split paper text into sections.
        
        Args:
            text: Full paper text
            
        Returns:
            Dictionary mapping section names to content.
            Keys are lowercase: 'abstract', 'introduction', etc.
        """
        if not text or not text.strip():
            return {}
        
        section_markers = self._find_section_markers(text)
        
        if not section_markers:
            return self._fallback_split(text)
        
        sections = {}
        lines = text.split('\n')
        
        sorted_markers = sorted(section_markers, key=lambda x: x[1])
        
        for i, (section_name, start_line) in enumerate(sorted_markers):
            if section_name == 'references':
                continue
            
            if i + 1 < len(sorted_markers):
                end_line = sorted_markers[i + 1][1]
            else:
                end_line = len(lines)
            
            content_lines = lines[start_line + 1:end_line]
            content = '\n'.join(content_lines).strip()
            
            if content:
                sections[section_name] = content
        
        return sections
    
    def _find_section_markers(self, text: str) -> list[tuple[str, int]]:
        """Find all section header positions in the text."""
        markers = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            
            for pattern, section_name in self._compiled_patterns:
                if pattern.match(line_stripped):
                    markers.append((section_name, line_num))
                    break
            else:
                match = self.NUMBERED_SECTION_PATTERN.match(line_stripped)
                if match:
                    section_title = match.group(2).lower().strip()
                    for _, name in self.SECTION_PATTERNS:
                        if name in section_title or section_title in name:
                            markers.append((name, line_num))
                            break
        
        return markers
    
    def _fallback_split(self, text: str) -> dict[str, str]:
        """
        Fallback splitting when no clear section headers found.
        Uses heuristics to identify abstract and main content.
        """
        sections = {}
        
        abstract_match = re.search(
            r'(?i)abstract[:\s]*\n(.*?)(?=\n\n|\n[A-Z]|\n\d+\.)',
            text,
            re.DOTALL
        )
        
        if abstract_match:
            sections['abstract'] = abstract_match.group(1).strip()
        
        paragraphs = text.split('\n\n')
        if paragraphs:
            if 'abstract' not in sections and len(paragraphs[0]) < 2000:
                sections['abstract'] = paragraphs[0].strip()
            
            main_content = '\n\n'.join(paragraphs[1:]) if 'abstract' in sections else text
            if main_content.strip():
                sections['main'] = main_content.strip()
        
        return sections
    
    def get_section(self, text: str, section_name: str) -> Optional[str]:
        """
        Get a specific section from the paper.
        
        Args:
            text: Full paper text
            section_name: Name of section to extract (lowercase)
            
        Returns:
            Section content or None if not found.
        """
        sections = self.split(text)
        return sections.get(section_name.lower())
    
    def extract_claims_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Get sections most likely to contain extractable claims.
        
        Returns sections in priority order for claim extraction:
        1. Results - empirical findings
        2. Abstract - key claims summary
        3. Discussion - interpretations
        4. Conclusion - final claims
        5. Methods - methodological claims
        
        Args:
            text: Full paper text
            
        Returns:
            List of (section_name, content) tuples in priority order.
        """
        sections = self.split(text)
        
        priority_order = ['results', 'abstract', 'discussion', 'conclusion', 'methods', 'introduction', 'main']
        
        result = []
        for section_name in priority_order:
            if section_name in sections and sections[section_name]:
                result.append((section_name, sections[section_name]))
        
        for name, content in sections.items():
            if name not in priority_order and content:
                result.append((name, content))
        
        return result


def split_into_paragraphs(text: str, min_length: int = 50) -> list[str]:
    """
    Split text into paragraphs for processing.
    
    Args:
        text: Text to split
        min_length: Minimum paragraph length to include
        
    Returns:
        List of paragraphs.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    
    result = []
    for para in paragraphs:
        para = para.strip()
        if len(para) >= min_length:
            result.append(para)
    
    return result


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.
    Rough approximation: ~4 characters per token for English text.
    """
    return len(text) // 4
