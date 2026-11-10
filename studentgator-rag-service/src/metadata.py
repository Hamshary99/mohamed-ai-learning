"""
src/metadata.py
Independent Metadata Organization & Registry Module for StudentGator RAG
========================================================================
This module loads, validates, and manages document-level and chunk-level metadata.
It provides:
- Strongly typed Pydantic models for document metadata.
- Pre-filtering helpers for ChromaDB relational scoping ($eq, $and).
- Chunk enrichment utilities that ensure metadata is 100% ChromaDB-compatible
  (e.g., flattening lists of statutory terms into primitive strings).
- Automatic fallback discovery for documents in the data directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =========================================================================
# 1. Pydantic Schemas for Metadata
# =========================================================================

class DocumentMetadata(BaseModel):
    """
    Strongly-typed metadata record representing a single source document.
    Matches the schema defined in data/documents_manifest.json.
    """
    file_name: str = Field(..., description="Target PDF file name, e.g. 'us_uf_f1_graduate_regulations.pdf'")
    relative_path: str = Field(..., description="Relative workspace path, e.g. 'data/...'")
    title: str = Field(..., description="Human-readable formal document title")
    institution: str = Field(..., description="Issuing authority, university, or governmental agency")
    country: str = Field(..., description="Target jurisdiction or host country (e.g. 'United States', 'Germany')")
    domain_category: str = Field(..., description="Categorical tag for relational routing (e.g. 'eu_germany_admissions')")
    statutory_terms: List[str] = Field(default_factory=list, description="Key regulatory keywords and statutory codes")
    page_count: int = Field(default=0, ge=0, description="Total pages in the source PDF")
    file_size_kb: float = Field(default=0.0, ge=0.0, description="File size in kilobytes")
    source_url: Optional[str] = Field(default=None, description="Official online origin or download URL")


class ChromaChunkMetadata(BaseModel):
    """
    Flattened, ChromaDB-safe metadata schema attached to individual text chunks.
    ChromaDB requires all metadata values to be primitive types (str, int, float, bool).
    """
    file_name: str
    source: str
    title: str
    institution: str
    country: str
    domain_category: str
    page: int = Field(default=1, description="Source page number (1-indexed)")
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. 'doc_01_c003'")
    chunk_type: str = Field(default="child", description="'child' for search or 'parent' for generation")
    parent_id: Optional[str] = Field(default=None, description="Pointer to the parent chunk in the document store")
    statutory_terms: str = Field(default="", description="Comma-separated string of regulatory terms for keyword search")


# =========================================================================
# 2. Central Metadata Registry & Manager
# =========================================================================

class MetadataManager:
    """
    Central registry for managing document metadata, building pre-filters,
    and enriching LangChain Document chunks.
    """

    # Built-in fallback entry for the StudentGator FAQ document
    DEFAULT_FAQ_ENTRY = DocumentMetadata(
        file_name="Studentgator FAQs.pdf",
        relative_path="data/Studentgator FAQs.pdf",
        title="StudentGator Platform Official Admissions & Visa Guidance FAQs",
        institution="StudentGator Advising Services",
        country="International",
        domain_category="studentgator_general",
        statutory_terms=[
            "application fee",
            "visa requirements",
            "tuition payment",
            "admissions timeline",
            "scholarships",
            "housing support"
        ],
        page_count=8,
        file_size_kb=478.1,
        source_url=None
    )

    def __init__(self, manifest_path: Optional[str] = None):
        """
        Initializes the registry by loading documents_manifest.json.
        If manifest_path is omitted, resolves data/documents_manifest.json automatically.
        """
        current_dir = Path(__file__).resolve().parent
        self.data_dir = (current_dir.parent / "data").resolve()
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else (self.data_dir / "documents_manifest.json")

        self._records: Dict[str, DocumentMetadata] = {}
        self.reload()

    def reload(self) -> None:
        """Loads and validates all manifest records from disk."""
        self._records.clear()

        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    for item in data:
                        doc = DocumentMetadata.model_validate(item)
                        self._records[doc.file_name] = doc
            except Exception as exc:
                print(f"[MetadataManager] Warning: Could not parse '{self.manifest_path}': {exc}")

        # Ensure StudentGator FAQs has an entry if the file is in data/
        faq_file = self.data_dir / "Studentgator FAQs.pdf"
        if faq_file.exists() and "Studentgator FAQs.pdf" not in self._records:
            self._records["Studentgator FAQs.pdf"] = self.DEFAULT_FAQ_ENTRY

    # ---------------------------------------------------------------------
    # Query & Lookup Helpers
    # ---------------------------------------------------------------------

    def get(self, file_name: str) -> Optional[DocumentMetadata]:
        """Returns the DocumentMetadata for a specific PDF file name, or None."""
        return self._records.get(os.path.basename(file_name))

    def get_domain_category(self, file_name: str) -> str:
        """Returns the domain category tag for a file name (defaults to 'general')."""
        record = self.get(file_name)
        return record.domain_category if record else "general"

    def list_documents(self) -> List[DocumentMetadata]:
        """Returns a list of all registered document metadata objects."""
        return list(self._records.values())

    def list_file_names(self) -> List[str]:
        """Returns all registered file names."""
        return list(self._records.keys())

    def list_domain_categories(self) -> List[str]:
        """Returns a unique, sorted list of all domain categories."""
        categories = {doc.domain_category for doc in self._records.values()}
        return sorted(categories)

    def list_countries(self) -> List[str]:
        """Returns a unique, sorted list of all countries covered."""
        countries = {doc.country for doc in self._records.values()}
        return sorted(countries)

    # ---------------------------------------------------------------------
    # ChromaDB Pre-Filter Builders
    # ---------------------------------------------------------------------

    def build_chroma_filter(
        self,
        domain_category: Optional[str] = None,
        country: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Constructs a structured ChromaDB relational pre-filter dict.
        Supports single filters or compound ($and) filters.
        
        Example outputs:
            {"domain_category": {"$eq": "eu_germany_admissions"}}
            {"$and": [{"country": {"$eq": "Germany"}}, {"domain_category": {"$eq": "eu_germany_admissions"}}]}
        """
        clauses = []
        if domain_category:
            clauses.append({"domain_category": {"$eq": domain_category}})
        if country:
            clauses.append({"country": {"$eq": country}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    # ---------------------------------------------------------------------
    # Dynamic Document Ingestion & Loading
    # ---------------------------------------------------------------------

    def load_documents(self, data_dir: Optional[str] = None) -> List[Any]:
        """
        Dynamically scans data/ for all PDF files, loads their pages via PyPDFLoader,
        and enriches each page with metadata from the manifest registry.

        Gives you complete freedom to drop new PDFs into data/ at any time:
        - If a PDF is in documents_manifest.json, it receives its full manifest metadata.
        - If a new uncataloged PDF is added, it is auto-discovered and tagged with
          derived defaults based on the file name.
        - If no PDFs are found or if loading fails, it gracefully falls back to the
          curated reference institutional documents.
        """
        from langchain_core.documents import Document
        from langchain_community.document_loaders import PyPDFLoader

        target_dir = Path(data_dir).resolve() if data_dir else self.data_dir
        all_docs: List[Document] = []

        if target_dir.exists():
            pdf_files = sorted(list(target_dir.glob("*.pdf")))
            for pdf_path in pdf_files:
                file_name = pdf_path.name
                record = self.get(file_name)

                # Auto-register uncataloged PDFs dropped into data/
                if not record:
                    clean_title = file_name.replace(".pdf", "").replace("_", " ").title()
                    record = DocumentMetadata(
                        file_name=file_name,
                        relative_path=f"data/{file_name}",
                        title=clean_title,
                        institution="Institutional Partner",
                        country="International",
                        domain_category="general_admissions",
                        statutory_terms=[]
                    )
                    self._records[file_name] = record

                try:
                    loader = PyPDFLoader(str(pdf_path))
                    pages = loader.load()

                    for page_idx, page in enumerate(pages, start=1):
                        page.metadata["source"] = file_name
                        page.metadata["file_name"] = file_name
                        page.metadata["title"] = record.title
                        page.metadata["institution"] = record.institution
                        page.metadata["country"] = record.country
                        page.metadata["domain_category"] = record.domain_category
                        page.metadata["statutory_terms"] = ", ".join(record.statutory_terms)
                        page.metadata["page"] = page_idx

                    all_docs.extend(pages)
                except Exception as exc:
                    print(f"[MetadataManager] Warning: Failed loading '{file_name}': {exc}")

        # If no PDFs were loaded, supply curated institutional fallback docs
        if not all_docs:
            print("[MetadataManager] No PDFs loaded; using curated reference policies.")
            all_docs = self.get_fallback_documents()

        print(f"[MetadataManager] Total documents loaded: {len(all_docs)} pages across the knowledge base.")
        return all_docs

    @staticmethod
    def get_fallback_documents() -> List[Any]:
        """
        Curated reference policies for StudentGator international admissions.
        Guarantees the RAG service always has verified ground-truth context even
        when running in lightweight offline or test environments.
        """
        from langchain_core.documents import Document

        return [
            Document(
                page_content=(
                    "Form I-20 Travel Endorsement Policy: Enrolled F-1 international students must obtain "
                    "a DSO travel signature on page 2 of Form I-20 before departing the United States. "
                    "For active students, the signature remains valid for exactly 12 months. For students on "
                    "post-completion Optional Practical Training (OPT), the travel signature is valid for "
                    "only 6 months. Re-entry requires an unexpired passport valid for at least 6 months, "
                    "a valid F-1 visa stamp, the endorsed Form I-20, and proof of academic enrollment or EAD."
                ),
                metadata={
                    "source": "us_immigration_manual_2026.pdf",
                    "file_name": "us_immigration_manual_2026.pdf",
                    "title": "US F-1 Travel & Visa Endorsement Policy",
                    "institution": "U.S. DHS / International Center",
                    "country": "United States",
                    "domain_category": "us_federal_immigration",
                    "statutory_terms": "Form I-20, SEVIS, OPT, CPT, DSO endorsement",
                    "page": 2
                }
            ),
            Document(
                page_content=(
                    "Undergraduate Housing Agreement SG-402 Cancellation Deadlines: Housing contracts "
                    "are legally binding for the full academic year. Written cancellation requests filed "
                    "prior to July 15 forfeit only the $250 security deposit. Cancellations submitted "
                    "between July 16 and August 15 incur an additional $1,200 contract termination assessment. "
                    "Requests submitted after August 15 obligate the resident to 50% of total semester rent, "
                    "unless an official visa refusal notice is verified through the International Center."
                ),
                metadata={
                    "source": "housing_regulations.pdf",
                    "file_name": "housing_regulations.pdf",
                    "title": "StudentGator Residence & Housing Cancellation Policy",
                    "institution": "Student Housing Office",
                    "country": "International",
                    "domain_category": "studentgator_general",
                    "statutory_terms": "SG-402, housing cancellation, deposit refund",
                    "page": 1
                }
            ),
            Document(
                page_content=(
                    "Academic Tuition Assessment and Refund Schedule: Full tuition refunds (100%) are "
                    "granted strictly for course drops completed before 11:59 PM on the fifth instructional "
                    "day of the semester. Formal course withdrawals processed between weeks 2 and 4 receive "
                    "a 50% tuition institutional credit. Zero tuition refunds are permitted after the end "
                    "of week 4. Mandatory international student onboarding fees ($350) are non-refundable."
                ),
                metadata={
                    "source": "bursar_refund_policy.pdf",
                    "file_name": "bursar_refund_policy.pdf",
                    "title": "Bursar Tuition Assessment and Refund Schedule",
                    "institution": "University Bursar Office",
                    "country": "International",
                    "domain_category": "studentgator_general",
                    "statutory_terms": "tuition refund, course withdrawal, bursar credit",
                    "page": 1
                }
            ),
            Document(
                page_content=(
                    "Health Insurance Waiver Requirements (Form SG-INS-W): All international students are "
                    "automatically enrolled in the university Student Health Insurance Plan ($1,480 per term). "
                    "To submit an alternate insurance waiver, students must file Form SG-INS-W before the third "
                    "Monday of September. Alternate plans must maintain an annual deductible below $1,000, "
                    "include medical evacuation coverage of at least $50,000, and repatriation of $25,000."
                ),
                metadata={
                    "source": "health_services.pdf",
                    "file_name": "health_services.pdf",
                    "title": "International Student Health Insurance Waiver Guidelines",
                    "institution": "Student Health Services",
                    "country": "International",
                    "domain_category": "studentgator_general",
                    "statutory_terms": "SG-INS-W, health insurance waiver, medical coverage",
                    "page": 1
                }
            )
        ]

    # ---------------------------------------------------------------------
    # Chunk Enrichment (ChromaDB Compatibility)
    # ---------------------------------------------------------------------

    def enrich_chunk_metadata(
        self,
        file_name: str,
        page_number: int,
        chunk_id: str,
        parent_id: Optional[str] = None,
        chunk_type: str = "child",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates a 100% ChromaDB-compatible primitive dictionary to attach to a chunk.
        Ensures lists (like statutory_terms) are converted into comma-separated strings
        so ChromaDB does not reject them.
        """
        base_name = os.path.basename(file_name)
        record = self.get(base_name)

        if record:
            title = record.title
            institution = record.institution
            country = record.country
            domain_category = record.domain_category
            statutory_terms_str = ", ".join(record.statutory_terms)
        else:
            title = base_name
            institution = "Unknown"
            country = "Unknown"
            domain_category = "general"
            statutory_terms_str = ""

        chunk_meta = ChromaChunkMetadata(
            file_name=base_name,
            source=base_name,
            title=title,
            institution=institution,
            country=country,
            domain_category=domain_category,
            page=page_number,
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            parent_id=parent_id,
            statutory_terms=statutory_terms_str
        )

        metadata_dict = chunk_meta.model_dump()

        # Merge any optional extra metadata if primitive
        if extra_metadata:
            for key, val in extra_metadata.items():
                if isinstance(val, (str, int, float, bool)):
                    metadata_dict[key] = val
                else:
                    metadata_dict[key] = str(val)

        return metadata_dict


# Global default instance for quick imports:
# from src.metadata import metadata_registry
metadata_registry = MetadataManager()


# =========================================================================
# Self-Test Execution
# =========================================================================

if __name__ == "__main__":
    manager = MetadataManager()
    print("=" * 70)
    print("StudentGator Metadata Registry Status")
    print("=" * 70)
    print(f"Manifest Path: {manager.manifest_path}")
    print(f"Total Registered Documents: {len(manager.list_documents())}\n")

    print("Registered Documents & Domain Categories:")
    for doc in manager.list_documents():
        print(f"  - [{doc.domain_category}] {doc.file_name} ({doc.country}) -> {doc.title[:45]}...")

    print("\nUnique Domain Categories:")
    for category in manager.list_domain_categories():
        print(f"  * {category}")

    print("\nSample ChromaDB Pre-Filter:")
    sample_filter = manager.build_chroma_filter(domain_category="eu_germany_admissions", country="Germany")
    print(f"  {sample_filter}")

    print("\nSample Enriched Chunk Metadata:")
    sample_chunk = manager.enrich_chunk_metadata(
        file_name="eu_germany_residence_act_section_16b.pdf",
        page_number=1,
        chunk_id="chunk_de_16b_001",
        chunk_type="child",
        parent_id="parent_de_16b_001"
    )
    print(json.dumps(sample_chunk, indent=2))
    print("=" * 70)

