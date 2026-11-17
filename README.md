# TalentRank

TalentRank is a Python application for analyzing and ranking resumes against
job-function-specific criteria. It includes a web interface and a command-line
pipeline.

> **Important:** Resume data is sensitive personal information. Use authorized
> data only, protect uploaded files and generated results, and review applicable
> privacy and employment regulations. Treat rankings as decision support—not
> as a substitute for human review.

## Features

- Extracts information from `.docx` resumes.
- Builds candidate features and reports confidence and missing-data indicators.
- Ranks candidates using configurable weighting and ranking components.
- Provides feature-based explanations and HR feedback endpoints.
- Reports evaluation metrics using HR ratings when available; otherwise, it
  uses a signal-based proxy. Proxy metrics are not a substitute for evaluation
  against independently reviewed HR ratings.

Supported functions include branding, e-commerce, human resources,
manufacturing, R&D, and sales.

## Requirements

- Python 3.10 or later (verify against your deployment environment).
- The dependencies listed in `code/requirements.txt`.

## Run the web application

```bash
cd code
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --reload
```

Open the application at <http://127.0.0.1:8000>. FastAPI's interactive API
documentation is available at <http://127.0.0.1:8000/docs>.

## Run the command-line pipeline

Place `.docx` resumes in a directory, then run from `code/`:

```bash
python main.py --function 品牌 --data_dir /path/to/resumes
```

Replace `品牌` with a supported function and provide the directory containing
the resumes. Use `python main.py --help` to see the available options.

## API overview

- `GET /api/health` — health check
- `POST /api/analyze` — analyze uploaded `.docx` resumes
- `POST /api/hr_rating` — submit HR ratings
- `POST /api/feedback` — submit candidate feedback
- `GET /docs` — interactive API documentation

## Evaluation

When HR gold-standard ratings are available, the application uses them for
evaluation. Otherwise, it calculates metrics from a feature-based proxy.
Check the returned `metrics_source` field before interpreting the metrics;
proxy-based results should not be presented as independently validated
ranking quality.

## Security and deployment

The development server is not intended for public production use. Before
processing real resumes or exposing the application to a network, review
authentication, access controls, upload limits, storage and retention of
personal data, and the API's cross-origin settings.
