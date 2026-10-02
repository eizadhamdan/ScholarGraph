# Neo4j interface & Cypher execution
import argparse
import json
from neo4j import GraphDatabase
from .config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


class GraphUnavailableError(RuntimeError):
    pass


class GraphQueryError(RuntimeError):
    pass


def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        connection_timeout=5.0,
    )


def check_graph_connection() -> bool:
    driver = None
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        return True
    except Exception:
        return False
    finally:
        if driver is not None:
            driver.close()


def import_triples_to_neo4j(json_path: str):
    """Hydrates local Neo4j instance with nodes and relationships from JSON."""
    driver = get_neo4j_driver()

    with open(json_path, "r") as f:
        graph_data = json.load(f)

    print(f"Connecting to Neo4j at {NEO4J_URI} and creating constraints...")
    with driver.session() as session:
        # Create schema constraints
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Concept) REQUIRE m.name IS UNIQUE"
        )

        print(f"Ingesting graph payload for {len(graph_data)} papers...")

        query = """
        UNWIND $batch AS item
        MERGE (p:Paper {id: item.paper_id})
        ON CREATE SET p.title = item.title

        // 1. Create Author Nodes & Relationships
        FOREACH (author IN item.authors |
            MERGE (a:Author {name: author})
            MERGE (a)-[:AUTHORED]->(p)
        )

        // 2. Create Category Nodes & Relationships
        FOREACH (cat IN item.categories |
            MERGE (c:Category {name: cat})
            MERGE (p)-[:IN_CATEGORY]->(c)
        )

        // 3. Create Concept Nodes & Relationships
        FOREACH (concept IN item.concepts |
            MERGE (m:Concept {name: concept})
            MERGE (p)-[:USES_METHOD]->(m)
        )
        """

        # Execute in batches of 500
        batch_size = 500
        for i in range(0, len(graph_data), batch_size):
            batch = graph_data[i : i + batch_size]
            session.run(query, batch=batch)

    driver.close()
    print("Neo4j database hydration complete!")


def run_cypher_query(cypher: str, parameters: dict | None = None) -> list[dict]:
    """Executes a Cypher query against Neo4j and returns dict results."""
    driver = None
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            return session.execute_read(
                lambda transaction: [
                    record.data()
                    for record in transaction.run(cypher, parameters or {})
                ]
            )
    except Exception as error:
        if getattr(error, "code", "").startswith("Neo.ClientError.Statement"):
            raise GraphQueryError(
                "Neo4j rejected the generated Cypher query."
            ) from error
        raise GraphUnavailableError(
            "Neo4j is unavailable or the graph query failed."
        ) from error
    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neo4j management.")
    parser.add_argument(
        "--import",
        dest="import_path",
        type=str,
        help="Path to graph triples JSON",
    )
    args = parser.parse_args()
    if args.import_path:
        import_triples_to_neo4j(args.import_path)
