"""
Candidate database indexed by AADHAAR number.
HR enters Aadhaar → system pulls record → compares with HR-entered details.
"""
import random
import hashlib

# ─── Official Records (indexed by Aadhaar) ───────────────────────────────────
AADHAAR_RECORDS = {
    "1234-5678-9012": {
        "name": "Rahul Sharma",
        "dob": "1992-05-14",
        "address": "42 Marine Lines, Mumbai, MH 400001",
        "prev_addresses": ["12 Andheri West, Mumbai, MH 400058"],
        "id_type": "Aadhaar",
        "id_expiry": "2030-01-01",
        "criminal_records": [], "sex_offender": False, "interpol": False,
        "credit_score": 780, "fraud_indicators": "None",
        "sanctions": False, "bankruptcy": False, "pep": False,
        "adverse_media": False, "litigation": False,
    },
    "9876-5432-1098": {
        "name": "Sneha Iyer",
        "dob": "1995-07-30",
        "address": "23 Indiranagar, Bengaluru, KA 560038",
        "prev_addresses": ["56 Koramangala, Bengaluru, KA 560034"],
        "id_type": "Aadhaar",
        "id_expiry": "2035-01-01",
        "criminal_records": [], "sex_offender": False, "interpol": False,
        "credit_score": 820, "fraud_indicators": "None",
        "sanctions": False, "bankruptcy": False, "pep": False,
        "adverse_media": False, "litigation": False,
    },
    "1111-2222-3333": {
        "name": "Priya Patel",
        "dob": "1988-11-22",
        "address": "7 Koregaon Park, Pune, MH 411001",
        "prev_addresses": ["15 FC Road, Pune, MH 411004"],
        "id_type": "Aadhaar",
        "id_expiry": "2023-06-15",   # expired
        "criminal_records": [{"type": "Misdemeanor", "year": 2015, "jurisdiction": "Mumbai Sessions Court", "status": "Acquitted"}],
        "sex_offender": False, "interpol": False,
        "credit_score": 620, "fraud_indicators": "Low",
        "sanctions": False, "bankruptcy": True, "pep": False,
        "adverse_media": True, "litigation": False,
    },
    "4444-5555-6666": {
        "name": "Amit Verma",
        "dob": "1985-03-08",
        "address": "88 Connaught Place, Delhi, DL 110001",
        "prev_addresses": [],
        "id_type": "Aadhaar",
        "id_expiry": "2035-01-01",
        "criminal_records": [{"type": "Fraud", "year": 2019, "jurisdiction": "Delhi High Court", "status": "Convicted"}],
        "sex_offender": False, "interpol": True,
        "credit_score": 480, "fraud_indicators": "High",
        "sanctions": True, "bankruptcy": False, "pep": True,
        "adverse_media": True, "litigation": True,
    },
}

# ─── Demo candidates helper for UI autofill ───────────────────────────────────
DEMO_CANDIDATES = {
    "Rahul Sharma":  {"aadhaar": "1234-5678-9012", "dob": "1992-05-14", "address": "42 Marine Lines, Mumbai, MH 400001"},
    "Sneha Iyer":    {"aadhaar": "9876-5432-1098", "dob": "1995-07-30", "address": "23 Indiranagar, Bengaluru, KA 560038"},
    "Priya Patel":   {"aadhaar": "1111-2222-3333", "dob": "1988-11-22", "address": "7 Koregaon Park, Pune, MH 411001"},
    "Amit Verma":    {"aadhaar": "4444-5555-6666", "dob": "1985-03-08", "address": "88 Connaught Place, Delhi, DL 110001"},
}

# ─── Dynamic generator for unknown Aadhaar numbers ───────────────────────────
NAMES = ["Arjun Singh", "Meera Nair", "Karan Mehta", "Divya Reddy", "Rohit Das",
         "Ananya Gupta", "Vikram Joshi", "Pooja Shah", "Suresh Kumar", "Lakshmi Rao"]
ID_TYPES = ["Aadhaar"]
CITIES = [
    ("Mumbai, MH 400001", "MH"), ("Delhi, DL 110001", "DL"),
    ("Bengaluru, KA 560001", "KA"), ("Chennai, TN 600001", "TN"),
    ("Hyderabad, TS 500001", "TS"), ("Pune, MH 411001", "MH"),
]
STREETS = ["12 MG Road", "45 Park Street", "7 Civil Lines", "33 Nehru Nagar",
           "89 Gandhi Road", "21 Sector 15", "5 Lake View Colony", "67 Ring Road"]
CRIME_TYPES = ["Traffic Violation", "Minor Fraud", "Trespass", "Public Disturbance"]
COURTS = ["City Civil Court", "District Court", "Sessions Court", "Magistrate Court"]


def _seed(aadhaar: str) -> random.Random:
    seed = int(hashlib.md5(aadhaar.encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def generate_record(aadhaar: str) -> dict:
    """Generate a deterministic fake record for any unknown Aadhaar number."""
    rng = _seed(aadhaar)
    roll = rng.random()

    name = rng.choice(NAMES)
    city, _ = rng.choice(CITIES)
    street = rng.choice(STREETS)
    address = f"{street}, {city}"

    # DOB: random year between 1970-2000
    yr = rng.randint(1970, 2000)
    mo = rng.randint(1, 12)
    dy = rng.randint(1, 28)
    dob = f"{yr}-{mo:02d}-{dy:02d}"

    # Expiry
    if rng.random() < 0.10:
        expiry = "2022-01-01"
    else:
        expiry = f"{rng.randint(2026, 2035)}-{rng.randint(1,12):02d}-01"

    credit_score = rng.randint(480, 850)

    if roll < 0.70:
        return {"name": name, "dob": dob, "address": address, "prev_addresses": [],
                "id_type": "Aadhaar", "id_expiry": expiry,
                "criminal_records": [], "sex_offender": False, "interpol": False,
                "credit_score": max(650, credit_score), "fraud_indicators": "None",
                "sanctions": False, "bankruptcy": False, "pep": False,
                "adverse_media": False, "litigation": False, "_generated": True}
    elif roll < 0.90:
        crimes = [{"type": rng.choice(CRIME_TYPES), "year": rng.randint(2010, 2022),
                   "jurisdiction": rng.choice(COURTS),
                   "status": rng.choice(["Acquitted", "Fine Paid", "Case Closed"])}] if rng.random() > 0.5 else []
        return {"name": name, "dob": dob, "address": address, "prev_addresses": [],
                "id_type": "Aadhaar", "id_expiry": expiry,
                "criminal_records": crimes, "sex_offender": False, "interpol": False,
                "credit_score": rng.randint(580, 680), "fraud_indicators": "Low",
                "sanctions": False, "bankruptcy": rng.random() > 0.7, "pep": False,
                "adverse_media": rng.random() > 0.6, "litigation": False, "_generated": True}
    else:
        return {"name": name, "dob": dob, "address": address, "prev_addresses": [],
                "id_type": "Aadhaar", "id_expiry": "2021-06-01",
                "criminal_records": [{"type": rng.choice(["Fraud", "Assault", "Money Laundering"]),
                                       "year": rng.randint(2015, 2023),
                                       "jurisdiction": rng.choice(COURTS),
                                       "status": rng.choice(["Convicted", "Under Trial"])}],
                "sex_offender": False, "interpol": rng.random() > 0.5,
                "credit_score": rng.randint(350, 520), "fraud_indicators": "High",
                "sanctions": rng.random() > 0.5, "bankruptcy": True,
                "pep": rng.random() > 0.7, "adverse_media": True,
                "litigation": True, "_generated": True}


def lookup_by_aadhaar(aadhaar: str) -> dict:
    """Look up record by Aadhaar number. Always returns a record."""
    aadhaar = aadhaar.strip()
    if aadhaar in AADHAAR_RECORDS:
        return AADHAAR_RECORDS[aadhaar]
    return generate_record(aadhaar)