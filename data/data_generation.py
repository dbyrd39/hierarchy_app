#!/usr/bin/env python3
"""
Generate a synthetic product master dataset for hierarchy testing.

- 10,000 rows total
- 200 total columns:
    * product_id
    * product_name
    * category_name
    * supplier_name
    * description
    * attr_001 ... attr_195  (sparse attributes, ~90% sparse overall)

- 200 meaningful category names (Pens, Notebooks, Shipping Boxes, Coffee Pods, etc.)
- Hybrid data:
    * Numeric-with-unit strings (e.g., "210 mm", "12.5 cm")
    * Categorical single values
    * Multi-value lists ("blue; black; red")
    * Boolean-ish flags
    * Messy free text
"""

from typing import List, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

NUM_ROWS = 10_000
NUM_ATTRS = 195         # attr_001 ... attr_195
NUM_CATEGORIES = 200
RANDOM_SEED = 42

# Approximate base sparsity for attributes
GLOBAL_PRESENCE_PROB = 0.30  # 30% non-null, so ~70% sparsity


# ---------------------------------------------------------------------
# Category list: 200 meaningful names
# ---------------------------------------------------------------------

def build_category_list(n: int) -> List[str]:
    """
    Return 200 meaningful category names with real-world / office context,
    instead of synthetic labels like 'Category_001'.
    """
    category_list = [
        # Office & Stationery (1–40)
        "Pens", "Pencils", "Markers", "Highlighters", "Notebooks", "Journals",
        "Sketchbooks", "Binders", "Folders", "Envelopes", "File Pockets",
        "Tab Dividers", "Sticky Notes", "Index Cards", "Clipboards",
        "Legal Pads", "Paper Reams", "Cardstock", "Labels",
        "Laminating Sheets", "Whiteboards", "Bulletin Boards", "Push Pins",
        "Paper Clips", "Binder Clips", "Rubber Bands", "Stamp Pads",
        "Ink Bottles", "Toner Cartridges", "Desk Calendars",
        "Wall Calendars", "Desk Organizers", "Drawer Organizers",
        "Desktop Trays", "Project Planners", "Presentation Folders",
        "Report Covers", "Signage Holders", "Brochure Stands", "Easel Pads",

        # Mailing / Packaging (41–80)
        "Shipping Boxes", "Padded Mailers", "Bubble Wrap", "Packing Peanuts",
        "Packing Tape", "Tape Dispensers", "Shipping Labels", "Postal Scales",
        "Address Labels", "Kraft Paper Rolls", "Shipping Tubes", "Pallet Wrap",
        "Stretch Film", "Strapping Kits", "Twine & Cord", "Zip Bags",
        "Poly Bags", "Shrink Bags", "Gift Bags", "Tissue Paper", "Tag Labels",
        "Hang Tags", "Chipboard Sheets", "Mailing Boxes", "Corrugated Sheets",
        "Carton Inserts", "Void Fill Rolls", "Foam Sheets", "Label Protectors",
        "Tape Dots", "Thermal Labels", "Pallet Corner Guards", "Steel Straps",
        "Nylon Straps", "Parcel Markers", "Delivery Pouches", "Return Kits",
        "Bag Sealers", "Industrial Tape", "Fragile Stickers",

        # Furniture & Workspace (81–120)
        "Office Chairs", "Drafting Chairs", "Task Chairs", "Standing Desks",
        "Desk Lamps", "Floor Lamps", "Monitor Stands", "Monitor Arms",
        "Laptop Risers", "Filing Cabinets", "Mobile Carts", "Bookcases",
        "Desk Hutches", "Drawer Units", "Conference Tables", "Training Tables",
        "Whiteboard Stands", "Coat Racks", "Foot Rests", "Keyboard Trays",
        "Anti-Fatigue Mats", "Waste Bins", "Recycling Bins", "Magazine Racks",
        "Desk Partitions", "Cubicle Panels", "Shelving Units", "Storage Bins",
        "Lockers", "Cash Drawers", "Desk Pads", "Chair Mats", "Table Lamps",
        "Workbenches", "Utility Carts", "Book Trolleys", "Desk Shelves",
        "Monitor Shelves", "Storage Drawers", "Literature Sorters",

        # Technology & Electronics (121–160)
        "Keyboards", "Mice", "Mouse Pads", "USB Drives",
        "External Hard Drives", "Surge Protectors", "HDMI Cables",
        "Ethernet Cables", "Charging Stations", "Webcams", "Headsets",
        "Bluetooth Speakers", "Label Printers", "Laser Printers",
        "Inkjet Printers", "Thermal Printers", "Shredders", "Fax Machines",
        "Calculators", "Projectors", "Projector Screens", "Document Cameras",
        "Wireless Adapters", "Smart Plugs", "Digital Pens", "Barcode Scanners",
        "Card Readers", "POS Terminals", "Paper Shredders", "Battery Packs",
        "Storage Servers", "Patch Panels", "Fiber Cables", "Toner Waste Bins",
        "Cooling Fans", "Docking Stations", "Laptop Bags", "Tablet Holders",
        "Phone Stands", "Web Switches",

        # Breakroom & Janitorial (161–200)
        "Coffee Pods", "Tea Bags", "Disposable Cups", "Disposable Plates",
        "Paper Towels", "Napkins", "Hand Soap", "Sanitizing Wipes",
        "Trash Liners", "Cleaning Sprays", "Air Fresheners", "Dish Soap",
        "Plastic Utensils", "Sugar Packets", "Creamer Cups", "Water Bottles",
        "First Aid Kits", "Safety Gloves", "Eye Protection", "Earplugs",
        "Dust Masks", "Brooms", "Mops", "Vacuum Bags", "Floor Cleaner",
        "Hand Towels", "Sponges", "Scrub Brushes", "Disinfectant Sprays",
        "Gloves", "Thermometers", "Aprons", "Hair Nets", "Plastic Wrap",
        "Aluminum Foil", "Storage Containers", "Food Bins",
        "Water Filters", "Breakroom Snacks", "Paper Placemats",
    ]

    if len(category_list) != n:
        raise ValueError(f"Expected {n} categories, but have {len(category_list)}")

    return category_list


CATEGORIES = build_category_list(NUM_CATEGORIES)

SUPPLIERS = [
    "Acme Supplies Co.",
    "Global Office Depot",
    "Northwind Distributors",
    "Contoso Wholesale",
    "StapleSource LLC",
    "PaperTrail Inc.",
]

COLORS = [
    "black", "blue", "red", "green", "yellow",
    "purple", "orange", "white", "clear",
]

MATERIALS = [
    "plastic", "metal", "wood", "poly", "cardboard",
    "vinyl", "mesh", "fabric", "leatherette",
]

ADJECTIVES = [
    "premium", "economy", "heavy-duty", "lightweight",
    "ergonomic", "refillable", "eco-friendly", "stackable",
    "compact", "large-capacity",
]

ITEM_TYPES = [
    "ballpoint pen", "gel pen", "rollerball pen", "binder",
    "notebook", "sticky notes", "shipping box", "envelope",
    "desk chair", "monitor stand", "keyboard", "mouse",
    "copy paper", "tab divider", "filing folder",
]

UNITS = ["mm", "cm", "inch", "g", "kg"]

MESSY_NULLS = ["", "N/A", "unknown", " ", None]


# ---------------------------------------------------------------------
# Attribute Column Schema
# ---------------------------------------------------------------------

def build_attribute_schema(num_attrs: int) -> List[Dict]:
    """
    Assign each attribute:
    - A type (numeric_unit, single_choice, multi_choice, flag, freetext)
    - 1–5 favored categories (randomly chosen from 200)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    schema: List[Dict] = []
    for i in range(1, num_attrs + 1):
        name = f"attr_{i:03d}"

        attr_type = rng.choice(
            ["numeric_unit", "single_choice", "multi_choice", "flag", "freetext"],
            p=[0.25, 0.25, 0.20, 0.15, 0.15],
        )

        # 1–5 favored categories for this attribute
        k = int(rng.integers(1, 6))
        favored_categories = rng.choice(CATEGORIES, size=k, replace=False).tolist()

        schema.append(
            {
                "name": name,
                "type": attr_type,
                "favored_categories": favored_categories,
            }
        )
    return schema


ATTR_SCHEMA = build_attribute_schema(NUM_ATTRS)


# ---------------------------------------------------------------------
# Attribute Value Generators
# ---------------------------------------------------------------------

def gen_numeric_with_unit(rng: np.random.Generator) -> str:
    base = float(rng.uniform(1, 1000))
    unit = rng.choice(UNITS)
    style = rng.integers(0, 3)
    if style == 0:
        return f"{base:.1f} {unit}"
    elif style == 1:
        return f"{int(round(base))}{unit}"
    else:
        return f"{base:.0f} {unit} "  # trailing space


def gen_single_choice(rng: np.random.Generator) -> str:
    pool = COLORS + MATERIALS + [
        "A4", "A5", "letter", "legal", "wide-ruled", "college-ruled",
        "glossy", "matte", "laminated",
    ]
    return str(rng.choice(pool))


def gen_multi_choice(rng: np.random.Generator) -> str:
    pool = COLORS + MATERIALS + [
        "recycled", "acid-free", "waterproof", "magnetic",
        "with lid", "with handles",
    ]
    k = int(rng.integers(2, 5))
    values = rng.choice(pool, size=k, replace=False)
    sep_style = rng.integers(0, 3)
    if sep_style == 0:
        return "; ".join(values)
    elif sep_style == 1:
        return ", ".join(values)
    else:
        return ";".join(values)


def gen_flag(rng: np.random.Generator) -> str:
    return rng.choice(["yes", "no", "true", "false", "Y", "N"])


def gen_freetext(rng: np.random.Generator) -> str:
    tokens = rng.choice(
        ADJECTIVES + COLORS + MATERIALS +
        ["bulk pack", "single", "3-pack", "10-pack"],
        size=int(rng.integers(3, 8)),
        replace=True,
    )
    return " ".join(tokens)


def gen_attribute_value(attr_type: str, rng: np.random.Generator) -> str:
    if attr_type == "numeric_unit":
        return gen_numeric_with_unit(rng)
    if attr_type == "single_choice":
        return gen_single_choice(rng)
    if attr_type == "multi_choice":
        return gen_multi_choice(rng)
    if attr_type == "flag":
        return gen_flag(rng)
    return gen_freetext(rng)


# ---------------------------------------------------------------------
# Product Metadata Generators
# ---------------------------------------------------------------------

def gen_product_id(idx: int) -> str:
    return f"SKU{idx:07d}"


def gen_product_name(rng: np.random.Generator) -> str:
    adj = rng.choice(ADJECTIVES)
    item = rng.choice(ITEM_TYPES)
    color = rng.choice(COLORS)
    return f"{adj.title()} {color.title()} {item.title()}"


def gen_description(rng, category, supplier, product_name):
    base = f"{product_name} from {supplier}. "
    dims = (
        f"Package width: {rng.integers(50, 400)} mm, "
        f"Package depth: {rng.integers(50, 400)} mm. "
    )
    features = rng.choice(
        [
            "Ideal for everyday office use.",
            "Perfect for home, school, or office.",
            "Designed for long-lasting performance.",
            "Great for organizing important documents.",
            "Suitable for high-volume printing.",
        ],
        size=int(rng.integers(1, 3)),
        replace=False,
    )
    return base + dims + " ".join(features)


# ---------------------------------------------------------------------
# Main Data Generator
# ---------------------------------------------------------------------

def generate_synthetic_product_master(
    num_rows: int = NUM_ROWS,
    num_attrs: int = NUM_ATTRS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    # Attribute presence probabilities per category
    attr_presence_probs: Dict[str, Dict[str, float]] = {}
    for spec in ATTR_SCHEMA:
        col_name = spec["name"]
        favored = set(spec["favored_categories"])
        probs: Dict[str, float] = {}
        for cat in CATEGORIES:
            if cat in favored:
                probs[cat] = min(0.25, GLOBAL_PRESENCE_PROB * 2.5)
            else:
                probs[cat] = max(0.02, GLOBAL_PRESENCE_PROB * 0.3)
        attr_presence_probs[col_name] = probs

    rows = []

    for i in range(num_rows):
        category = rng.choice(CATEGORIES)
        supplier = rng.choice(SUPPLIERS)

        row = {
            "product_id": gen_product_id(i + 1),
            "product_name": gen_product_name(rng),
            "category_name": category,
            "supplier_name": supplier,
            "description": gen_description(rng, category, supplier, ""),
        }

        # 195 sparse attributes
        for spec in ATTR_SCHEMA:
            col = spec["name"]
            attr_type = spec["type"]
            presence_prob = attr_presence_probs[col][category]

            if rng.random() < presence_prob:
                # Real or messy value
                if rng.random() < 0.95:
                    value = gen_attribute_value(attr_type, rng)
                else:
                    value = rng.choice(MESSY_NULLS)
            else:
                # Mostly missing
                if rng.random() < 0.9:
                    value = np.nan
                else:
                    value = rng.choice(MESSY_NULLS)

            row[col] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    # Enforce column order
    fixed_cols = [
        "product_id",
        "product_name",
        "category_name",
        "supplier_name",
        "description",
    ]
    attr_cols = [f"attr_{i:03d}" for i in range(1, NUM_ATTRS + 1)]
    df = df[fixed_cols + attr_cols]

    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    df = generate_synthetic_product_master()
    print("Generated:", df.shape)

    attr_cols = [c for c in df.columns if c.startswith("attr_")]
    non_null_frac = 1 - df[attr_cols].isna().mean().mean()
    print(f"Approximate non-null fraction in attribute columns: {non_null_frac:.1%}")

    output_path = "synthetic_product_master_200_categories_literary.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
