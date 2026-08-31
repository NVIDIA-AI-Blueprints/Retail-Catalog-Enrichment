"""Synthetic retail catalog generator.

Produces the input the whole pipeline reads:
    product_id (string, required), title (string), description (string),
    metadata (string, optional free-form)

Three design notes that matter for the benchmark, not just for realism:

* **Messiness is the workload.** Real catalogs carry vendor SKU noise in titles, HTML
  fragments, mojibake, doubled whitespace, ALL CAPS, and empty descriptions. A clean
  synthetic catalog would understate both the input token count and the difficulty of
  extraction, making every downstream number optimistic.

* **Length is long-tailed on purpose.** A mean input length over a uniform distribution
  is a fiction. Descriptions here span roughly 0 to 1200 characters with a heavy right
  tail, so the in-flight batcher sees the same ragged shape it will see in production.

* **Deterministic.** Same `--seed`, same bytes. The benchmark compares configurations, so
  the catalog must not be a variable.

Swap this for your own catalog as soon as you have one — everything downstream reads the
Parquet, not the generator.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------------------
# Vocabulary.
#
# Hand-built rather than pulled from `faker`, so output is byte-identical across machines
# and needs no network. Breadth matters more than depth: a catalog that exercises three of
# the schema's category enums would not test the grammar's branching.
# --------------------------------------------------------------------------------------

BRANDS = [
    "Brand01", "Brand02", "Brand03", "Brand04", "Brand05", "Brand06", "Brand07",
    "Brand08", "Brand09", "Brand10", "Brand11", "Brand12", "Brand13", "Brand14",
    "Brand15", "Brand16", "Brand17", "Brand18", "Brand19", "Brand20",
    "GENERIC", "OEM",
]

# (category, [product nouns], [material words], size vocabulary kind, [feature tags])
# Feature tags select which FEATURES_BY_TAG buckets a family may draw from; "generic"
# is appended to every family.
PRODUCT_FAMILIES = [
    ("apparel", ["Crewneck Sweatshirt", "Flannel Shirt", "Chino Pant", "Puffer Vest",
                 "Merino Base Layer", "Linen Blazer", "Rain Shell", "Cargo Short"],
     ["cotton", "merino wool", "polyester blend", "linen", "recycled nylon"], "clothing",
     ["textile"]),
    ("footwear", ["Trail Runner", "Chelsea Boot", "Canvas Sneaker", "Wool Slipper",
                  "Hiking Boot", "Leather Loafer"],
     ["full-grain leather", "suede", "canvas", "knit mesh", "rubber"], "shoe",
     ["textile"]),
    ("accessories", ["Leather Belt", "Wool Beanie", "Canvas Tote", "Dopp Kit",
                     "Silk Scarf", "Bifold Wallet"],
     ["leather", "wool", "waxed canvas", "silk"], "onesize", ["textile", "carry"]),
    ("home_kitchen", ["Cast Iron Skillet", "Stoneware Mug", "Chef Knife", "Cutting Board",
                      "French Press", "Mixing Bowl Set", "Dutch Oven"],
     ["cast iron", "stoneware", "high-carbon steel", "acacia wood", "borosilicate glass"],
     "volume", ["cookware"]),
    ("furniture", ["Oak Side Table", "Upholstered Armchair", "Floating Shelf",
                   "Bar Stool", "Writing Desk"],
     ["solid oak", "walnut veneer", "powder-coated steel", "boucle"], "dimension",
     ["hardware"]),
    ("beauty_personal_care", ["Beard Oil", "Clay Mask", "Shampoo Bar", "Hand Salve",
                              "Lip Balm"],
     ["shea butter", "kaolin clay", "jojoba oil"], "volume", ["consumable"]),
    ("electronics", ["Bluetooth Speaker", "USB-C Hub", "Mechanical Keyboard",
                     "Noise Cancelling Headphone", "Desk Lamp"],
     ["aluminum", "ABS plastic", "tempered glass"], "onesize", ["hardware", "carry"]),
    ("sports_outdoors", ["Yoga Mat", "Insulated Bottle", "Trekking Pole", "Dry Bag",
                         "Climbing Chalk"],
     ["TPE foam", "stainless steel", "ripstop nylon", "carbon fiber"], "volume",
     ["carry", "hardware"]),
    ("toys_games", ["Wooden Block Set", "Card Game", "Plush Bear", "Puzzle 1000pc"],
     ["birch plywood", "recycled paperboard", "organic cotton"], "onesize",
     ["hardware", "textile"]),
    ("grocery", ["Single Origin Coffee", "Olive Oil", "Hot Sauce", "Sea Salt Flakes",
                 "Honey"],
     [], "volume", ["consumable"]),
    ("pet_supplies", ["Rope Tug Toy", "Ceramic Pet Bowl", "Reflective Leash",
                      "Orthopedic Dog Bed"],
     ["cotton rope", "ceramic", "nylon webbing", "memory foam"], "clothing",
     ["textile", "cookware"]),
    ("office_supplies", ["Dot Grid Notebook", "Gel Pen Set", "Desk Organizer",
                         "Kraft Folder"],
     ["recycled paper", "bamboo", "steel mesh"], "onesize", ["hardware", "carry"]),
    ("automotive", ["Microfiber Towel Pack", "Phone Mount", "Tire Gauge",
                    "All-Weather Floor Mat"],
     ["microfiber", "ABS plastic", "rubber"], "onesize", ["hardware"]),
]

COLORS = [
    "Black", "Charcoal", "Heather Grey", "Navy", "Olive", "Rust", "Cream", "Sand",
    "Forest", "Burgundy", "Slate Blue", "Mustard", "Terracotta", "Ivory", "Sage",
]

SIZE_VOCAB = {
    "clothing": [["XS", "S", "M", "L", "XL"], ["S", "M", "L"], ["M", "L", "XL", "XXL"]],
    "shoe": [["8", "9", "10", "11", "12"], ["7", "8", "9", "10"]],
    "volume": [["8 oz"], ["12 oz"], ["500 ml"], ["1 L"], ["250 ml"]],
    "dimension": [['24" x 18"'], ['36" x 20" x 29"']],
    "onesize": [[], ["One Size"]],
}

# Features are bucketed by tag and families declare which tags apply to them. Without
# this, a tire gauge ends up "seasoned at the foundry" -- messy input is wanted, but
# *incoherent* input is not: this catalog is also the substrate for the accuracy eval,
# and you cannot score an extraction against a listing that contradicts itself.
FEATURES_BY_TAG: dict[str, list[str]] = {
    "textile": [
        "reinforced double stitching at every stress point",
        "machine washable and rated for repeated commercial laundering",
        "pre-shrunk so the fit you buy is the fit you keep after the first wash",
        "moisture wicking construction that pulls sweat away from the skin",
        "oeko-tex certified, meaning every component has been tested for harmful substances",
        "naturally antimicrobial, which keeps odor down between washes",
        "quick drying, so it packs away damp without going musty overnight",
        "fade resistant finish that holds its color through seasons of direct sun",
    ],
    "cookware": [
        "dishwasher safe on the top rack, though hand washing preserves the finish longer",
        "hand wash only; prolonged soaking will damage the finish",
        "seasoned at the foundry and ready to use straight out of the box",
        "food safe and free of the coatings that flake off cheaper alternatives",
        "BPA free and independently tested for food contact safety",
    ],
    "hardware": [
        "ships flat packed and assembles in under fifteen minutes with the included hardware",
        "compatible with standard mounts and most third-party accessories",
        "fade resistant finish that holds its color through seasons of direct sun",
        "wipe-clean surface that shrugs off fingerprints",
    ],
    "carry": [
        "packable design that compresses down to roughly the size of a water bottle",
        "designed for everyday carry, with a profile that disappears in a jacket pocket",
        "includes a drawstring storage pouch for travel and off-season storage",
        "quick drying, so it packs away damp without going musty overnight",
    ],
    "consumable": [
        "BPA free and independently tested for food contact safety",
        "food safe and free of the coatings that flake off cheaper alternatives",
        "small-batch production with a best-by date printed on every unit",
    ],
    "generic": [
        "sourced from certified suppliers audited annually for labor practices",
        "backed by a limited lifetime warranty against manufacturing defects",
    ],
}

# Longer prose blocks. Real listings carry spec paragraphs, use-case copy and boilerplate;
# these are what push the ISL distribution into its realistic range, and the tail of that
# distribution is what the in-flight batcher actually has to cope with.
SPEC_BLOCKS = [
    "Dimensions are approximate and measured flat: length {a} inches, width {b} inches, "
    "height {c} inches. Weight is roughly {w} grams before packaging. Please allow for "
    "slight variation between production runs, as each batch is finished by hand.",

    "Construction details: the outer shell is bonded to an inner liner using a low-VOC "
    "adhesive, the seams are taped rather than glued, and every unit is inspected before "
    "it leaves the facility. Hardware is rated to {w} grams of static load.",

    "Fit notes: this style runs slightly generous through the body. If you are between "
    "sizes we recommend sizing down for a closer fit, or staying true to size if you plan "
    "to layer underneath. Inseam measures {b} inches on the mid size and is graded across "
    "the run.",

    "Care and longevity: with routine maintenance this piece is designed to last years "
    "rather than seasons. Store away from direct sunlight, avoid harsh solvents, and "
    "address stains promptly rather than letting them set.",
]

USE_CASE_BLOCKS = [
    "Equally at home on a commute as it is on a weekend trip. Customers tell us they "
    "bought one, then came back for a second within the month, which is the only product "
    "review metric we really pay attention to.",

    "We designed this around a simple complaint: everything in this category is either "
    "overbuilt and heavy or cheap and disposable. This sits deliberately in the middle -- "
    "durable enough to abuse, light enough to actually carry.",

    "Works as well for a first-time buyer as for someone replacing a worn-out favorite. "
    "The materials are forgiving, the maintenance is minimal, and there is nothing here "
    "that needs a manual.",
]

BOILERPLATE_BLOCKS = [
    "Shipping: orders placed before 2pm local time ship same day. Free standard shipping "
    "on orders over $50. Expedited options are available at checkout.",

    "Returns: unworn items may be returned within 30 days in original packaging for a full "
    "refund. Final sale items are marked as such on the product page.",

    "Please note that colors may appear differently across displays. If the shade is "
    "critical to your purchase, contact us and we will describe it against a reference.",
]

MARKETING_FLUFF = [
    "A wardrobe staple you'll reach for again and again.",
    "Built to last, designed to be used.",
    "Our best seller, back in stock.",
    "Thoughtfully made in small batches.",
    "The upgrade you didn't know you needed.",
    "Perfect gift for the holidays!!",
    "Customers love this one.",
    "Simple, durable, honest.",
]

CARE = [
    "Machine wash cold, tumble dry low.", "Hand wash and dry immediately.",
    "Wipe clean with a damp cloth.", "Do not bleach.",
    "Season with oil after each use.", "Dishwasher safe, top rack.",
]

ORIGINS = ["Made in Portugal", "Made in Vietnam", "Made in USA", "Imported",
           "Made in Italy", "Assembled in Mexico"]

# Non-English stragglers. Real catalogs acquired through M&A or marketplace feeds always
# have some; they exercise both the tokenizer (multi-byte -> more tokens per character)
# and the model's willingness to still emit English-normalized attributes.
FOREIGN_SNIPPETS = [
    "Envío gratis en pedidos superiores a 50€. Material de alta calidad.",
    "Livraison rapide. Fabriqué à partir de matériaux durables.",
    "Hochwertige Verarbeitung. Pflegeleicht und langlebig.",
    "高品質な素材を使用しています。",
    "Frete grátis para todo o Brasil.",
]

HTML_WRAPPERS = [
    "<p>{}</p>", "<div class='prod-desc'>{}</div>", "{}<br/><br/>",
    "<span style=\"font-size:12px\">{}</span>", "<ul><li>{}</li></ul>",
]

# Deliberately ambiguous characters: this is what a real feed's encoding
# damage looks like, and reproducing it is the point.
MOJIBAKE = ["â€™", "Ã©", "â€œ", "â€\x9d", "Â®", "&amp;", "&nbsp;", " "]  # noqa: RUF001


# --------------------------------------------------------------------------------------
# Row construction
# --------------------------------------------------------------------------------------


def _noisy_title(rng: random.Random, brand: str, noun: str, color: str,
                 sizes: list[str]) -> str:
    """Build a title with the kind of noise a real feed carries.

    Vendor SKU fragments, redundant color/size repetition and inconsistent casing are
    the three things that make `normalized_title` a non-trivial extraction rather than
    a string copy.
    """
    parts = []
    if rng.random() < 0.85:
        parts.append(brand)
    parts.append(noun)
    if rng.random() < 0.55:
        parts.append(f"- {color}")
    if sizes and rng.random() < 0.35:
        parts.append(f"({'/'.join(sizes[:3])})")
    if rng.random() < 0.40:
        sku = f"{rng.choice('ABCDEFGHJKMNPQRSTVWXYZ')}{rng.randint(100, 999)}" \
              f"-{rng.randint(10, 99)}"
        parts.append(rng.choice([f"[{sku}]", f"SKU:{sku}", f"#{sku}", sku]))
    if rng.random() < 0.12:
        parts.append(rng.choice(["NEW", "SALE", "**HOT**", "Free Shipping", "2-PACK"]))

    title = " ".join(parts)

    roll = rng.random()
    if roll < 0.08:
        title = title.upper()
    elif roll < 0.14:
        title = title.lower()
    if rng.random() < 0.10:
        title = title.replace(" ", "  ")
    return title


def _description(rng: random.Random, noun: str, material: str, color: str,
                 sizes: list[str], tags: list[str]) -> str:
    """Long-tailed description text.

    The length distribution is the point, and it is the single generator decision that
    most affects every downstream number: ISL is the denominator of products/sec, so a
    catalog of uniformly short blurbs would produce a throughput figure that no real
    customer could reproduce.

    Four tiers, roughly matching what a scraped multi-vendor catalog looks like:
      ~8%  no description at all -- forces enrichment from the title alone
      ~25% thin: a line or two from a lazy vendor feed
      ~50% typical: a feature list plus a paragraph
      ~17% rich: full spec + use case + boilerplate, and this tail sets the mean
    """
    r = rng.random()
    if r < 0.08:
        return ""

    fmt = {"a": rng.randint(4, 40), "b": rng.randint(3, 30),
           "c": rng.randint(1, 20), "w": rng.randint(50, 4000)}

    parts: list[str] = [f"The {noun} in {color}."]
    if material:
        parts.append(f"Made from {material}.")

    if r < 0.33:            # thin
        n_features, n_spec, n_use, n_boiler = rng.randint(1, 2), 0, 0, 0
    elif r < 0.83:          # typical
        n_features, n_spec, n_use, n_boiler = rng.randint(3, 5), 1, rng.randint(0, 1), 0
    else:                   # rich -- the tail
        n_features = rng.randint(6, 10)
        n_spec, n_use, n_boiler = rng.randint(1, 3), rng.randint(1, 2), rng.randint(1, 3)

    pool = sorted({p for t in [*tags, "generic"] for p in FEATURES_BY_TAG[t]})
    parts.extend(rng.sample(pool, k=min(n_features, len(pool))))
    parts.extend(b.format(**fmt) for b in rng.sample(SPEC_BLOCKS, k=n_spec))
    parts.extend(rng.sample(USE_CASE_BLOCKS, k=n_use))
    parts.extend(rng.sample(BOILERPLATE_BLOCKS, k=n_boiler))

    if rng.random() < 0.5:
        parts.append(rng.choice(MARKETING_FLUFF))
    if rng.random() < 0.35:
        parts.append(rng.choice(CARE))
    if rng.random() < 0.30:
        parts.append(rng.choice(ORIGINS) + ".")
    if sizes and rng.random() < 0.4:
        parts.append("Available in " + ", ".join(sizes) + ".")
    if rng.random() < 0.06:
        parts.append(rng.choice(FOREIGN_SNIPPETS))

    text = " ".join(s if s.endswith(".") else s + "." for s in parts)

    if rng.random() < 0.30:
        text = rng.choice(HTML_WRAPPERS).format(text)
    if rng.random() < 0.15:
        for _ in range(rng.randint(1, 3)):
            pos = rng.randint(0, max(0, len(text) - 1))
            text = text[:pos] + rng.choice(MOJIBAKE) + text[pos:]
    if rng.random() < 0.10:
        text = "\n\n".join([text[: len(text) // 2], text[len(text) // 2:]])
    return text


def _metadata(rng: random.Random, brand: str, color: str, sizes: list[str],
              material: str) -> str:
    """Free-form metadata blob.

    Deliberately inconsistent in both shape and key naming across rows -- some JSON,
    some key=value, some empty. A pipeline that assumes a stable metadata schema is
    assuming something real catalogs do not provide.
    """
    r = rng.random()
    if r < 0.25:
        return ""
    fields = {}
    if rng.random() < 0.8:
        fields[rng.choice(["vendor", "supplier", "mfr"])] = brand
    if rng.random() < 0.6:
        fields[rng.choice(["colour", "color", "primary_color"])] = color
    if rng.random() < 0.5 and sizes:
        fields["sizes"] = "|".join(sizes)
    if rng.random() < 0.4 and material:
        fields[rng.choice(["material", "fabric", "composition"])] = material
    if rng.random() < 0.3:
        fields["weight_g"] = str(rng.randint(50, 4000))
    if rng.random() < 0.2:
        fields["condition"] = rng.choice(["new", "New", "NEW", "refurb", "open box"])
    if not fields:
        return ""
    if r < 0.65:
        return json.dumps(fields)
    return "; ".join(f"{k}={v}" for k, v in fields.items())


def _stated(value: str, listing: str) -> bool:
    """Did this value survive into the emitted text?

    A value the generator chose is only a valid label if the listing actually says it.
    A product whose description came out empty has no material stated anywhere, so the
    correct answer for `material` is null, not the word we happened to pick. Scoring a
    model against information absent from its input measures nothing but our bookkeeping.
    """
    return bool(value) and value.lower() in listing


def generate_rows(n: int, seed: int) -> tuple[dict[str, list], dict[str, list]]:
    """Return (catalog rows, ground truth rows).

    The generator necessarily knows the right answer -- it picks a category, brand,
    color, material and size set, then writes a messy listing *around* them. Discarding
    that would throw away a free labelled set, and customers judge quality as closely as
    they judge speed.

    The subtlety that makes this honest: a value the generator chose is only a valid
    label if it actually survived into the emitted text. A product whose description
    came out empty has no material stated anywhere, so the correct answer for `material`
    is null, not the word we happened to pick. Scoring a model against information that
    is not present in its input measures nothing but our own bookkeeping.

    So each field is checked for literal presence in the emitted text and recorded as
    null when absent. That also makes the null cases useful in their own right: they are
    where a model can be caught inventing an answer.
    """
    rng = random.Random(seed)
    ids, titles, descriptions, metadatas = [], [], [], []
    gt_category, gt_brand, gt_color, gt_material, gt_sizes = [], [], [], [], []

    for i in range(n):
        category, nouns, materials, size_kind, tags = rng.choice(PRODUCT_FAMILIES)
        noun = rng.choice(nouns)
        brand = rng.choice(BRANDS)
        color = rng.choice(COLORS)
        material = rng.choice(materials) if materials else ""
        sizes = rng.choice(SIZE_VOCAB[size_kind])

        # Zero-padded and sequential: `product_id` is the idempotency key for replay, so
        # it must be stable across regenerations.
        product_id = f"SKU-{i:08d}"
        title = _noisy_title(rng, brand, noun, color, sizes)
        description = _description(rng, noun, material, color, sizes, tags)
        metadata = _metadata(rng, brand, color, sizes, material)

        ids.append(product_id)
        titles.append(title)
        descriptions.append(description)
        metadatas.append(metadata)

        # Only label what the listing actually says.
        listing = f"{title}\n{description}\n{metadata}".lower()

        # Category is the exception: it is never named literally, but the product noun
        # determines it unambiguously and the noun is always in the title.
        gt_category.append(category)
        gt_brand.append(brand if _stated(brand, listing) else None)
        gt_color.append(color if _stated(color, listing) else None)
        gt_material.append(material if _stated(material, listing) else None)
        gt_sizes.append([s for s in sizes if _stated(s, listing)])

    catalog = {
        "product_id": ids,
        "title": titles,
        "description": descriptions,
        "metadata": metadatas,
    }
    truth = {
        "product_id": ids,
        "category": gt_category,
        "brand": gt_brand,
        "primary_color": gt_color,
        "material": gt_material,
        "sizes": gt_sizes,
    }
    return catalog, truth


# The input contract, declared explicitly rather than inferred from the data, so drift in
# either direction is a loud failure.
INPUT_SCHEMA = pa.schema([
    pa.field("product_id", pa.string(), nullable=False),
    pa.field("title", pa.string(), nullable=False),
    pa.field("description", pa.string(), nullable=True),
    pa.field("metadata", pa.string(), nullable=True),
])

# Ground truth goes in a SEPARATE file, never merged into the catalog. The input contract
# is what a customer hands the pipeline, and it must not acquire columns that exist only
# because our input happens to be synthetic.
GROUND_TRUTH_SCHEMA = pa.schema([
    pa.field("product_id", pa.string(), nullable=False),
    pa.field("category", pa.string(), nullable=False),
    pa.field("brand", pa.string(), nullable=True),
    pa.field("primary_color", pa.string(), nullable=True),
    pa.field("material", pa.string(), nullable=True),
    pa.field("sizes", pa.list_(pa.string()), nullable=False),
])


def write_shards(rows: dict[str, list], out_dir: Path, shards: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(rows, schema=INPUT_SCHEMA)

    n = table.num_rows
    per = (n + shards - 1) // shards
    written = []
    for s in range(shards):
        start = s * per
        if start >= n:
            break
        chunk = table.slice(start, min(per, n - start))
        path = out_dir / f"shard_{s:04d}.parquet"
        # Fixed row_group_size and no compression jitter: byte-level determinism is
        # what makes "same seed, same bytes" checkable with a diff.
        pq.write_table(chunk, path, row_group_size=2048, compression="snappy")
        written.append(path)
    return written


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--n", type=int, default=10_000, help="number of products")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed; same seed, same bytes")
    ap.add_argument("--out", type=Path, default=Path("data/catalog"),
                    help="output directory for Parquet shards")
    ap.add_argument("--shards", type=int, default=1,
                    help="split output across N Parquet files")
    ap.add_argument("--ground-truth", type=Path, default=None,
                    help="also write a labelled set here, for `catalog-bench accuracy`")


def main(args: argparse.Namespace) -> int:
    rows, truth = generate_rows(args.n, args.seed)
    paths = write_shards(rows, args.out, args.shards)

    n_empty_desc = sum(1 for d in rows["description"] if not d)
    lengths = sorted(len(t) + len(d)
                     for t, d in zip(rows["title"], rows["description"], strict=True))
    print(f"wrote {args.n} products to {len(paths)} shard(s) in {args.out}")
    print(f"  chars/product  p50={lengths[len(lengths) // 2]}  "
          f"p95={lengths[int(len(lengths) * 0.95)]}  max={lengths[-1]}")
    print(f"  empty descriptions: {n_empty_desc} ({n_empty_desc / args.n:.1%})")

    if args.ground_truth:
        args.ground_truth.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(truth, schema=GROUND_TRUTH_SCHEMA), args.ground_truth,
                       row_group_size=2048, compression="snappy")
        print(f"wrote ground truth to {args.ground_truth}")
        for field in ("brand", "primary_color", "material"):
            stated = sum(1 for v in truth[field] if v)
            print(f"  {field:<14} stated in {stated / args.n:>5.1%} of listings "
                  f"({args.n - stated} unrecoverable -> null)")
        n_sizes = sum(1 for v in truth["sizes"] if v)
        print(f"  {'sizes':<14} stated in {n_sizes / args.n:>5.1%} of listings")
    return 0
