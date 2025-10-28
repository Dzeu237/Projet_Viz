crypto-data-pipeline/
│
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── docker-compose.yml                 # Orchestration services
├── Makefile                           # Commandes automatisées
├── .env.example                       # Variables d'environnement
├── .gitignore
│
├── config/                            # Configuration
│   ├── airflow/                       # Config Airflow
│   │   ├── airflow.cfg
│   │   └── connections.json
│   ├── database/                      # Config bases de données
│   │   ├── postgres_init.sql
│   │   └── redis.conf
│   ├── monitoring/                    # Config monitoring
│   │   ├── prometheus.yml
│   │   └── grafana_dashboards.json
│   └── pipeline_config.yaml           # Config pipeline
│
├── airflow/                           # Apache Airflow
│   ├── dags/                          # DAGs Airflow
│   │   ├── __init__.py
│   │   ├── crypto_batch_daily.py      # Batch quotidien
│   │   ├── crypto_batch_hourly.py     # Batch horaire
│   │   ├── news_ingestion.py          # Ingestion news
│   │   ├── social_data_pipeline.py    # Données sociales
│   │   └── data_quality_checks.py     # Contrôles qualité
│   ├── plugins/                       # Plugins custom
│   │   ├── operators/
│   │   │   ├── crypto_api_operator.py
│   │   │   └── data_quality_operator.py
│   │   ├── sensors/
│   │   │   └── api_availability_sensor.py
│   │   └── hooks/
│   │       ├── crypto_api_hook.py
│   │       └── postgres_hook.py
│   └── logs/                          # Logs Airflow
│
├── src/                               # Code source
│   ├── __init__.py
│   │
│   ├── ingestion/                     # Ingestion données
│   │   ├── __init__.py
│   │   ├── base_ingester.py           # Classe de base
│   │   ├── stream/                    # Streaming
│   │   │   ├── __init__.py
│   │   │   ├── binance_stream.py      # WebSocket Binance
│   │   │   ├── coinbase_stream.py     # WebSocket Coinbase
│   │   │   └── stream_processor.py    # Traitement streams
│   │   ├── batch/                     # Batch
│   │   │   ├── __init__.py
│   │   │   ├── coingecko_batch.py     # API CoinGecko
│   │   │   ├── coinmarketcap_batch.py # API CoinMarketCap
│   │   │   └── historical_loader.py   # Données historiques
│   │   └── social/                    # Données sociales
│   │       ├── __init__.py
│   │       ├── reddit_scraper.py
│   │       ├── twitter_scraper.py
│   │       └── news_aggregator.py
│   │
│   ├── processing/                    # Traitement données
│   │   ├── __init__.py
│   │   ├── cleaners/                  # Nettoyage
│   │   │   ├── __init__.py
│   │   │   ├── price_cleaner.py
│   │   │   ├── text_cleaner.py
│   │   │   └── outlier_detector.py
│   │   ├── transformers/              # Transformations
│   │   │   ├── __init__.py
│   │   │   ├── price_transformer.py   # Prix normalisés
│   │   │   ├── technical_indicators.py # Indicateurs tech
│   │   │   └── sentiment_analyzer.py  # Analyse sentiment
│   │   ├── aggregators/               # Agrégations
│   │   │   ├── __init__.py
│   │   │   ├── time_aggregator.py     # Agrégations temporelles
│   │   │   └── multi_source_merger.py # Fusion sources
│   │   └── validators/                # Validation
│   │       ├── __init__.py
│   │       ├── schema_validator.py
│   │       └── data_quality_validator.py
│   │
│   ├── storage/                       # Couche stockage
│   │   ├── __init__.py
│   │   ├── postgres_manager.py        # PostgreSQL
│   │   ├── redis_manager.py           # Redis cache
│   │   ├── parquet_manager.py         # Fichiers Parquet
│   │   └── storage_factory.py         # Factory pattern
│   │
│   ├── quality/                       # Data quality
│   │   ├── __init__.py
│   │   ├── expectations.py            # Great Expectations
│   │   ├── anomaly_detection.py       # Détection anomalies
│   │   ├── drift_detection.py         # Détection drift
│   │   └── quality_metrics.py         # Métriques qualité
│   │
│   ├── monitoring/                    # Monitoring
│   │   ├── __init__.py
│   │   ├── metrics_collector.py       # Collecte métriques
│   │   ├── alerts.py                  # Système alertes
│   │   ├── health_checker.py          # Health checks
│   │   └── performance_tracker.py     # Tracking performance
│   │
│   ├── api/                           # API REST
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app
│   │   ├── routers/
│   │   │   ├── prices.py              # Endpoints prix
│   │   │   ├── indicators.py          # Endpoints indicateurs
│   │   │   ├── sentiment.py           # Endpoints sentiment
│   │   │   └── health.py              # Health check
│   │   ├── models/                    # Pydantic models
│   │   │   ├── requests.py
│   │   │   └── responses.py
│   │   └── dependencies.py
│   │
│   └── utils/                         # Utilitaires
│       ├── __init__.py
│       ├── logger.py                  # Logging configuré
│       ├── config.py                  # Config loader
│       ├── retry.py                   # Retry logic
│       └── helpers.py
│
├── streaming/                         # Services streaming
│   ├── consumer.py                    # Consommateur principal
│   ├── producer.py                    # Producteur
│   └── processors/
│       ├── real_time_aggregator.py
│       └── alert_processor.py
│
├── database/                          # Schémas base de données
│   ├── migrations/                    # Migrations Alembic
│   │   ├── versions/
│   │   │   ├── 001_initial_schema.py
│   │   │   ├── 002_add_sentiment.py
│   │   │   └── 003_add_indices.py
│   │   ├── env.py
│   │   └── alembic.ini
│   ├── schemas/
│   │   ├── bronze_layer.sql           # Raw data
│   │   ├── silver_layer.sql           # Cleaned data
│   │   └── gold_layer.sql             # Aggregated data
│   └── seeds/                         # Données de test
│       └── sample_data.sql
│
├── scripts/                           # Scripts utilitaires
│   ├── setup/
│   │   ├── init_database.sh           # Init DB
│   │   ├── create_tables.sh           # Créer tables
│   │   └── setup_airflow.sh           # Setup Airflow
│   ├── backfill/
│   │   ├── backfill_prices.py         # Backfill historique
│   │   └── backfill_social.py
│   ├── monitoring/
│   │   ├── check_pipeline_health.py
│   │   └── generate_metrics_report.py
│   └── maintenance/
│       ├── cleanup_old_data.py
│       └── optimize_tables.py
│
├── tests/                             # Tests
│   ├── __init__.py
│   ├── unit/                          # Tests unitaires
│   │   ├── test_ingestion/
│   │   ├── test_processing/
│   │   ├── test_storage/
│   │   └── test_quality/
│   ├── integration/                   # Tests intégration
│   │   ├── test_pipeline_e2e.py
│   │   ├── test_api_integration.py
│   │   └── test_database_integration.py
│   ├── performance/                   # Tests performance
│   │   └── test_throughput.py
│   └── fixtures/                      # Fixtures
│       ├── sample_data.json
│       └── mock_responses.json
│
├── monitoring/                        # Monitoring & Observabilité
│   ├── dashboards/                    # Dashboards Grafana
│   │   ├── pipeline_overview.json
│   │   ├── data_quality.json
│   │   ├── performance_metrics.json
│   │   └── alerts_dashboard.json
│   ├── prometheus/
│   │   └── rules.yml                  # Règles alertes
│   └── logs/                          # Logs centralisés
│
├── dashboard/                         # Dashboard Streamlit
│   ├── app.py                         # App principale
│   ├── pages/
│   │   ├── 01_real_time_prices.py     # Prix temps réel
│   │   ├── 02_technical_analysis.py   # Analyse technique
│   │   ├── 03_sentiment_analysis.py   # Analyse sentiment
│   │   ├── 04_correlations.py         # Corrélations
│   │   ├── 05_alerts.py               # Système alertes
│   │   └── 06_pipeline_status.py      # Status pipeline
│   ├── components/
│   │   ├── live_charts.py             # Graphiques live
│   │   ├── metrics_cards.py
│   │   └── alert_config.py
│   └── assets/
│
├── data/                              # Données (git-ignored)
│   ├── raw/                           # Bronze layer
│   │   ├── prices/
│   │   ├── social/
│   │   └── news/
│   ├── processed/                     # Silver layer
│   │   ├── cleaned_prices/
│   │   └── sentiment_scores/
│   ├── aggregated/                    # Gold layer
│   │   ├── daily_summaries/
│   │   └── indicators/
│   └── cache/                         # Cache Redis dumps
│
├── docker/                            # Docker configuration
│   ├── airflow/
│   │   └── Dockerfile
│   ├── api/
│   │   └── Dockerfile
│   ├── streaming/
│   │   └── Dockerfile
│   ├── dashboard/
│   │   └── Dockerfile
│   └── monitoring/
│       └── Dockerfile
│
├── docs/                              # Documentation
│   ├── architecture/
│   │   ├── system_design.md           # Design système
│   │   ├── data_flow.md               # Flux données
│   │   └── architecture_diagram.png
│   ├── guides/
│   │   ├── setup_guide.md             # Guide installation
│   │   ├── deployment_guide.md        # Guide déploiement
│   │   ├── user_guide.md              # Guide utilisateur
│   │   └── troubleshooting.md         # Résolution problèmes
│   ├── api/
│   │   └── api_documentation.md       # Doc API
│   ├── data/
│   │   ├── data_dictionary.md         # Dictionnaire données
│   │   └── schema_documentation.md    # Doc schémas
│   └── runbooks/
│       ├── incident_response.md
│       └── maintenance.md
│
├── notebooks/                         # Notebooks analyse
│   ├── 01_data_exploration.ipynb
│   ├── 02_quality_analysis.ipynb
│   └── 03_performance_analysis.ipynb
│
└── infrastructure/                    # Infrastructure as Code
    ├── terraform/                     # Terraform (si cloud)
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    └── kubernetes/                    # K8s manifests (si K8s)
        ├── deployments/
        ├── services/
        └── configmaps/
🏗️ Architecture
Vue d'ensemble
[APIs Crypto] ──┐
[Reddit/Twitter]─┼─> [Ingestion Layer] ─> [Processing] ─> [Storage]
[News Feeds] ────┘         │                   │             │
                           │                    │              ├─> [PostgreSQL]
                           │                    │              ├─> [Redis Cache]
                           │                    │              └─> [Parquet Files]
                           │                    │
                    [Data Quality]       [Monitoring]
                           │                    │
                           └────────┬───────────┘
                                    │
                              [API Layer]
                                    │
                         ┌──────────┴──────────┐
                         │                           │
                   [Dashboard]                    [Alertes]
Layers (Medallion Architecture)

Bronze (Raw) : Données brutes non transformées
Silver (Cleaned) : Données nettoyées et validées
Gold (Aggregated) : Données agrégées pour analyse

🚀 Quick Start
Prérequis
Docker & Docker Compose
Python 3.10+
PostgreSQL 14+
Redis 7+