"""
Generates a simple SVG logo for PneumoVision when no PNG logo exists.
"""


def get_logo_svg() -> str:
    """Return an inline SVG logo for PneumoVision."""
    return """
    <svg width="280" height="60" viewBox="0 0 280 60" xmlns="http://www.w3.org/2000/svg">
        <!-- Lung icon -->
        <g transform="translate(10, 5)">
            <!-- Left lung -->
            <path d="M20 8 C12 8 5 18 5 30 C5 42 10 48 18 48 C22 48 24 44 24 40 L24 12 C24 9 22 8 20 8Z"
                  fill="none" stroke="#00C49A" stroke-width="2.5" opacity="0.9"/>
            <!-- Right lung -->
            <path d="M30 8 C38 8 45 18 45 30 C45 42 40 48 32 48 C28 48 26 44 26 40 L26 12 C26 9 28 8 30 8Z"
                  fill="none" stroke="#00C49A" stroke-width="2.5" opacity="0.9"/>
            <!-- Trachea -->
            <line x1="25" y1="0" x2="25" y2="10" stroke="#00C49A" stroke-width="2.5"/>
            <!-- Cross-hatch for detail -->
            <line x1="12" y1="25" x2="22" y2="25" stroke="#00C49A" stroke-width="1" opacity="0.4"/>
            <line x1="28" y1="25" x2="38" y2="25" stroke="#00C49A" stroke-width="1" opacity="0.4"/>
            <line x1="14" y1="32" x2="22" y2="32" stroke="#00C49A" stroke-width="1" opacity="0.4"/>
            <line x1="28" y1="32" x2="36" y2="32" stroke="#00C49A" stroke-width="1" opacity="0.4"/>
        </g>
        <!-- Text -->
        <text x="65" y="38" font-family="Arial, Helvetica, sans-serif" font-size="28"
              font-weight="bold" fill="#FAFAFA">
            Pneumo<tspan fill="#00C49A">Vision</tspan>
        </text>
    </svg>
    """


def get_logo_html() -> str:
    """Return the logo as embeddable HTML."""
    svg = get_logo_svg()
    return f'<div style="text-align:center; margin-bottom:8px;">{svg}</div>'
