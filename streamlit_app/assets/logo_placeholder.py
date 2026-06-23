"""
Generates a simple SVG logo for PneumoVision when no PNG logo exists.
"""


def get_logo_svg() -> str:
    """Return an inline SVG logo for PneumoVision."""
    return """
    <svg width="240" height="50" viewBox="0 0 240 50" xmlns="http://www.w3.org/2000/svg">
        <!-- Lung icon -->
        <g transform="translate(5, 2)" opacity="0.95">
            <!-- Left lung -->
            <path d="M18 8 C11 8 5 16 5 26 C5 36 9 42 16 42 C20 42 22 38 22 35 L22 12 C22 9 20 8 18 8Z"
                  fill="none" stroke="#00C49A" stroke-width="2.5"/>
            <!-- Right lung -->
            <path d="M28 8 C35 8 41 16 41 26 C41 36 37 42 30 42 C26 42 24 38 24 35 L24 12 C24 9 26 8 28 8Z"
                  fill="none" stroke="#00C49A" stroke-width="2.5"/>
            <!-- Trachea -->
            <line x1="23" y1="0" x2="23" y2="10" stroke="#00C49A" stroke-width="2.5"/>
            <!-- Cross-hatch details -->
            <line x1="12" y1="22" x2="20" y2="22" stroke="#00C49A" stroke-width="1" opacity="0.3"/>
            <line x1="26" y1="22" x2="34" y2="22" stroke="#00C49A" stroke-width="1" opacity="0.3"/>
            <line x1="13" y1="28" x2="20" y2="28" stroke="#00C49A" stroke-width="1" opacity="0.3"/>
            <line x1="26" y1="28" x2="33" y2="28" stroke="#00C49A" stroke-width="1" opacity="0.3"/>
        </g>
        <!-- Text -->
        <text x="58" y="32" font-family="'Inter', system-ui, sans-serif" font-size="22"
              font-weight="700" fill="#FAFAFA" letter-spacing="-0.03em">
            Pneumo<tspan fill="#00C49A">Vision</tspan>
        </text>
    </svg>
    """


def get_logo_html() -> str:
    """Return the logo as embeddable HTML."""
    svg = get_logo_svg()
    return f'<div style="text-align:center; margin-bottom:8px;">{svg}</div>'
