#!/usr/bin/env python3
"""
Generate a synthetic product master dataset with:
- 10,000 rows
- 200 meaningful category names
- 195 structured attribute columns
- Realistic mixed-type sparse data
- Strong clustering signal for attribute-based clustering
"""

from typing import List, Dict
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

NUM_ROWS = 10_000
NUM_ATTRS = 195
NUM_CATEGORIES = 200
RANDOM_SEED = 42

GLOBAL_PRESENCE_PROB = 0.20  # improved sparsity (20% non-null)


# ---------------------------------------------------------------------
# CATEGORY LIST (200 meaningful names)
# ---------------------------------------------------------------------

def build_category_list(n: int) -> List[str]:
    category_list = [
        # Office & Stationery
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

        # Mailing / Packaging
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

        # Furniture & Workspace
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

        # Technology & Electronics
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

        # Breakroom & Janitorial
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
        raise ValueError(f"Expected {n} categories, got {len(category_list)}")

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
    "purple", "orange", "white", "clear"
]

MATERIALS = [
    "plastic", "metal", "wood", "poly", "cardboard",
    "vinyl", "mesh", "fabric"
]

ADJECTIVES = [
    "premium", "economy", "heavy-duty", "lightweight",
    "ergonomic", "refillable", "eco-friendly", "stackable",
    "compact", "large-capacity"
]

ITEM_TYPES = [
    "ballpoint pen", "gel pen", "rollerball pen", "binder",
    "notebook", "sticky notes", "shipping box", "envelope",
    "desk chair", "monitor stand", "keyboard", "mouse",
]

UNITS = ["mm", "cm", "inch", "g", "kg"]
MESSY_NULLS = ["", "N/A", "unknown", " ", None]


# ---------------------------------------------------------------------
# ATTRIBUTE DOMAIN → CATEGORY MAPPING
# ---------------------------------------------------------------------

DOMAIN_MAP = {
    "Writing": [
        "Pens", "Pencils", "Markers", "Highlighters", "Stamp Pads", "Ink Bottles"
    ],
    "Paper": [
        "Notebooks", "Journals", "Sketchbooks", "Sticky Notes",
        "Legal Pads", "Paper Reams", "Cardstock", "Index Cards"
    ],
    "Filing": [
        "Folders", "Binders", "File Pockets", "Report Covers",
        "Presentation Folders", "Drawer Units"
    ],
    "Mailing": [
        "Shipping Boxes", "Padded Mailers", "Bubble Wrap", "Shipping Labels"
    ],
    "Packaging": [
        "Packing Tape", "Tape Dots", "Foam Sheets", "Poly Bags", "Gift Bags"
    ],
    "Furniture": [
        "Office Chairs", "Task Chairs", "Standing Desks", "Bookcases", "Desk Organizers"
    ],
    "Technology": [
        "Keyboards", "Mice", "USB Drives", "Headsets", "Webcams", "Projectors"
    ],
    "Breakroom": [
        "Coffee Pods", "Tea Bags", "Disposable Cups", "Trash Liners", "Cleaning Sprays"
    ]
}

# reverse mapping
CATEGORY_TO_DOMAIN = {}
for domain, cats in DOMAIN_MAP.items():
    for c in cats:
        CATEGORY_TO_DOMAIN[c] = domain

DEFAULT_DOMAIN = "General"


# ---------------------------------------------------------------------
# ATTRIBUTE FAMILIES PER DOMAIN
# ---------------------------------------------------------------------

DOMAIN_ATTRIBUTES = {
    "Writing": [
        ("ink_color", "single_choice"),
        ("tip_size_mm", "numeric_unit"),
        ("pen_material", "single_choice"),
        ("refill_type", "single_choice"),
    ],
    "Paper": [
        ("sheet_count", "numeric_unit"),
        ("paper_weight_gsm", "numeric_unit"),
        ("paper_material", "single_choice"),
        ("page_rule_type", "single_choice"),
    ],
    "Filing": [
        ("folder_size", "single_choice"),
        ("spine_width_mm", "numeric_unit"),
        ("binder_ring_type", "single_choice"),
    ],
    "Mailing": [
        ("box_length_cm", "numeric_unit"),
        ("box_width_cm", "numeric_unit"),
        ("box_height_cm", "numeric_unit"),
        ("seal_type", "single_choice"),
    ],
    "Packaging": [
        ("material_type", "single_choice"),
        ("strength_rating", "single_choice"),
        ("closure_type", "single_choice"),
    ],
    "Furniture": [
        ("material", "single_choice"),
        ("weight_capacity_kg", "numeric_unit"),
        ("height_adjustable", "flag"),
        ("width_cm", "numeric_unit"),
    ],
    "Technology": [
        ("device_interface", "single_choice"),
        ("power_rating_w", "numeric_unit"),
        ("cable_type", "single_choice"),
        ("connectivity", "multi_choice"),
    ],
    "Breakroom": [
        ("flavor", "single_choice"),
        ("pack_size", "numeric_unit"),
        ("is_disposable", "flag"),
    ],
    "General": [
        ("material", "single_choice"),
        ("size_dimension", "freetext"),
    ]
}


# ---------------------------------------------------------------------
# BUILD STRUCTURED ATTRIBUTE SCHEMA
# ---------------------------------------------------------------------

def build_attribute_schema(num_attrs: int) -> List[Dict]:
    rng = np.random.default_rng(RANDOM_SEED)

    domains = list(DOMAIN_ATTRIBUTES.keys())
    schema = []

    for i in range(1, num_attrs + 1):
        name = f"attr_{i:03d}"

        # pick a domain
        domain = rng.choice(domains)
        family = DOMAIN_ATTRIBUTES[domain]

        # choose attribute template
        attr_name, attr_type = family[rng.integers(0, len(family))]

        # column name (structured)
        column_name = f"{name}_{attr_name}"

        # favored categories = categories in domain
        favored_categories = [
            c for c, d in CATEGORY_TO_DOMAIN.items()
            if d == domain
        ]
        if not favored_categories:
            favored_categories = CATEGORIES

        schema.append({
            "name": column_name,
            "type": attr_type,
            "favored_categories": favored_categories,
            "domain": domain,
        })

    return schema


ATTR_SCHEMA = build_attribute_schema(NUM_ATTRS)


# ---------------------------------------------------------------------
# ATTRIBUTE VALUE GENERATORS
# ---------------------------------------------------------------------

def gen_numeric_with_unit(rng):
    base = float(rng.uniform(1, 1000))
    unit = rng.choice(UNITS)
    return f"{base:.1f} {unit}"


def gen_single_choice(rng):
    pool = COLORS + MATERIALS + [
        "A4", "A5", "letter", "legal", "matte", "glossy"
    ]
    return str(rng.choice(pool))


def gen_multi_choice(rng):
    pool = COLORS + MATERIALS + ["recycled", "magnetic", "waterproof"]
    k = int(rng.integers(2, 4))
    values = rng.choice(pool, size=k, replace=False)
    return "; ".join(values)


def gen_flag(rng):
    return rng.choice(["yes", "no", "true", "false", "Y", "N"])


def gen_freetext(rng):
    tokens = rng.choice(
        ADJECTIVES + COLORS + MATERIALS,
        size=int(rng.integers(3, 7)),
        replace=True
    )
    return " ".join(tokens)


def gen_attribute_value(attr_type, rng):
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
# PRODUCT FIELDS
# ---------------------------------------------------------------------

def gen_product_id(i):
    return f"SKU{i:07d}"


def gen_product_name(rng):
    adj = rng.choice(ADJECTIVES)
    item = rng.choice(ITEM_TYPES)
    color = rng.choice(COLORS)
    return f"{adj.title()} {color.title()} {item.title()}"


def gen_description(rng, category, supplier):
    dims = (
        f"Package width: {rng.integers(50, 400)} mm, "
        f"Package depth: {rng.integers(50, 400)} mm. "
    )
    return f"{category} from {supplier}. {dims}"


# ---------------------------------------------------------------------
# MAIN GENERATOR
# ---------------------------------------------------------------------

def generate_synthetic_product_master():
    rng = np.random.default_rng(RANDOM_SEED)

    # build category → attribute presence probabilities
    attr_presence = {}
    for spec in ATTR_SCHEMA:
        col = spec["name"]
        domain = spec["domain"]

        favored = [
            c for c, d in CATEGORY_TO_DOMAIN.items()
            if d == domain
        ]

        probs = {}
        for cat in CATEGORIES:
            if cat in favored:
                probs[cat] = 0.35
            else:
                probs[cat] = 0.05

        attr_presence[col] = probs

    rows = []

    for i in range(NUM_ROWS):
        category = rng.choice(CATEGORIES)
        supplier = rng.choice(SUPPLIERS)

        row = {
            "product_id": gen_product_id(i),
            "product_name": gen_product_name(rng),
            "category_name": category,
            "supplier_name": supplier,
            "description": gen_description(rng, category, supplier),
        }

        # attributes
        for spec in ATTR_SCHEMA:
            col = spec["name"]
            attr_type = spec["type"]
            p = attr_presence[col][category]

            if rng.random() < p:
                if rng.random() < 0.95:
                    val = gen_attribute_value(attr_type, rng)
                else:
                    val = rng.choice(MESSY_NULLS)
            else:
                val = np.nan

            row[col] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    # enforce column order
    fixed = [
        "product_id", "product_name", "category_name",
        "supplier_name", "description"
    ]
    attr_cols = [spec["name"] for spec in ATTR_SCHEMA]
    df = df[fixed + attr_cols]

    return df


# ---------------------------------------------------------------------
# CLI EXECUTION
# ---------------------------------------------------------------------

if __name__ == "__main__":
    df = generate_synthetic_product_master()
    print(df.head())
    print(df.shape)

    out = "synthetic_product_master_200_categories_structured.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)


