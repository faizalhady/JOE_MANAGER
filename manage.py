"""
Vector DB Manager - CLI Entry Point

Usage:
    python manage.py init                    # Initialize database schema
    python manage.py ingest <file_or_dir>    # Ingest a file or directory
    python manage.py search "query text"     # Test similarity search
    python manage.py status                  # Show ingestion summary
    python manage.py stats                   # Show database statistics
    python manage.py delete <source_id>      # Delete a source and its chunks
    python manage.py purge                   # Delete ALL data (careful!)
    python manage.py config                  # Show current configuration
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import LOG_LEVEL, print_config


def setup_logging():
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_init():
    """Initialize the database schema."""
    from db.connection import DatabaseManager

    with DatabaseManager() as db:
        db.init_schema()
        print("\nDatabase schema initialized successfully.")
        print("Tables created: source_documents, document_chunks")
        print("Extension enabled: pgvector")


def cmd_ingest(path: str):
    """Ingest a file or directory."""
    from ingestion.pipeline import IngestionPipeline

    # Preserve UNC network paths — join remaining args in case of spaces
    target = Path(path)
    if not target.exists():
        print(f"Error: Path not found: {path}")
        print(f"  Tip: Wrap network paths in quotes:")
        print(f'  python manage.py ingest "\\\\server\\share\\folder"')
        sys.exit(1)

    pipeline = IngestionPipeline()
    pipeline.initialize()

    try:
        if target.is_file():
            source_id = pipeline.ingest_file(str(target))
            if source_id:
                print(f"\nFile ingested successfully (source_id={source_id})")
            else:
                print(f"\nFile skipped (already ingested or unsupported)")
        elif target.is_dir():
            results = pipeline.ingest_directory(str(target))
            print(f"\n{'='*50}")
            print(f"INGESTION SUMMARY")
            print(f"{'='*50}")
            print(f"  Processed: {results['processed']}")
            print(f"  Skipped:   {results['skipped']}")
            print(f"  Failed:    {results['failed']}")
            for f in results["files"]:
                status_icon = {"processed": "✓", "skipped": "⊘", "failed": "✗"}
                icon = status_icon.get(f["status"], "?")
                print(f"  {icon} {f['file']} - {f['status']}")
    finally:
        pipeline.shutdown()


def cmd_search(query: str, top_k: int = 5):
    """Test similarity search against the vector database."""
    from db.connection import DatabaseManager
    from embeddings.embedder import Embedder

    embedder = Embedder()
    embedder.load()

    # Use embed_query() — applies BGE instruction prefix for better retrieval
    query_vector = embedder.embed_query(query)

    with DatabaseManager() as db:
        # context_window=1 means: also fetch 1 chunk before and 1 chunk after each match
        results = db.search_with_context(query_vector, top_k=top_k, context_window=1)

        if not results:
            print("No results found. Is the database empty?")
            return

        print(f"\nSearch: \"{query}\"")
        print(f"{'='*60}")

        for i, r in enumerate(results, 1):
            similarity_pct = r["similarity"] * 100
            has_before = len(r.get('context_before', [])) > 0
            has_after = len(r.get('context_after', [])) > 0
            context_label = ""
            if has_before or has_after:
                context_label = f" [+{int(has_before)+int(has_after)} neighbor chunks]"

            print(f"\n--- Result {i} (similarity: {similarity_pct:.1f}%){context_label} ---")
            print(f"Source: {r['source_file']} ({r['source_type']})")
            print(f"Type:   {r['chunk_type']}")
            if r.get("metadata"):
                print(f"Meta:   {r['metadata']}")
            print(f"\nMatched chunk:")
            print(f"  {r['content']}")

            if r.get('context_before'):
                print(f"\n  [Context before]:")
                for ctx in r['context_before']:
                    preview = ctx[:300] + "..." if len(ctx) > 300 else ctx
                    print(f"  {preview}")

            if r.get('context_after'):
                print(f"\n  [Context after]:")
                for ctx in r['context_after']:
                    preview = ctx[:300] + "..." if len(ctx) > 300 else ctx
                    print(f"  {preview}")


def cmd_status():
    """Show ingestion summary."""
    from db.connection import DatabaseManager

    with DatabaseManager() as db:
        summary = db.get_ingestion_summary()

        if not summary:
            print("No documents ingested yet.")
            return

        print(f"\n{'='*70}")
        print(f"{'File':<30} {'Type':<6} {'Status':<12} {'Chunks':<8} {'Date'}")
        print(f"{'='*70}")

        for doc in summary:
            name = doc["file_name"][:29]
            print(f"{name:<30} {doc['file_type']:<6} {doc['status']:<12} "
                  f"{doc['total_chunks'] or 0:<8} {doc['created_at']}")


def cmd_stats():
    """Show database statistics."""
    from db.connection import DatabaseManager

    with DatabaseManager() as db:
        stats = db.get_stats()

        print(f"\n{'='*40}")
        print(f"VECTOR DATABASE STATISTICS")
        print(f"{'='*40}")
        print(f"  Total sources:      {stats['total_sources']}")
        print(f"  Completed:          {stats['completed_sources']}")
        print(f"  Failed:             {stats['failed_sources']}")
        print(f"  Total chunks:       {stats['total_chunks']}")
        print(f"  Embedding models:   {stats['embedding_models_used']}")


def cmd_delete(source_id: int):
    """Delete a source document and all its chunks."""
    from db.connection import DatabaseManager

    with DatabaseManager() as db:
        file_name = db.delete_source(source_id)
        if file_name:
            print(f"Deleted: {file_name} (id={source_id}) and all its chunks")
        else:
            print(f"No source found with id={source_id}")


def cmd_purge():
    """Delete ALL data from the database."""
    from db.connection import DatabaseManager

    confirm = input("WARNING: This will delete ALL vectors and source records. Type 'YES' to confirm: ")
    if confirm != "YES":
        print("Aborted.")
        return

    with DatabaseManager() as db:
        db.purge_all()
        print("All data purged.")


def cmd_config():
    """Print current configuration."""
    print_config()


# ─── CLI Router ───

def main():
    setup_logging()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "init":
        cmd_init()
    elif command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python manage.py ingest <file_or_directory>")
            print('  For paths with spaces: python manage.py ingest "\\\\server\\share\\my folder"')
            sys.exit(1)
        # Join all remaining args to handle paths with spaces
        # e.g. sys.argv = ['manage.py', 'ingest', '\\server\share\my', 'folder']
        ingest_path = " ".join(sys.argv[2:])
        cmd_ingest(ingest_path)
    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python manage.py search \"your query here\"")
            sys.exit(1)
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        cmd_search(sys.argv[2], top_k)
    elif command == "status":
        cmd_status()
    elif command == "stats":
        cmd_stats()
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python manage.py delete <source_id>")
            sys.exit(1)
        cmd_delete(int(sys.argv[2]))
    elif command == "purge":
        cmd_purge()
    elif command == "config":
        cmd_config()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
