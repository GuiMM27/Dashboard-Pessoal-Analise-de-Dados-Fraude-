"""Upload PUBLICO exclusivamente da pasta hf_export/ (sem dados brutos)."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Publicar o modelo educacional no Hugging Face Hub')
    parser.add_argument('--repo-id', required=True, help='usuario/nome-do-repositorio (ex.: meuuser/fraud-detection-ml)')
    args = parser.parse_args()
    if args.repo_id.count('/') != 1 or not all(args.repo_id.split('/')):
        parser.error('Use --repo-id usuario/nome-do-modelo')
    package_dir = ROOT / 'hf_export'
    expected = {'fraud_model.joblib', 'README.md', 'metadata.json', 'example_input.json', 'requirements-model.txt'}
    found = {f.name for f in package_dir.iterdir()} if package_dir.is_dir() else set()
    if found != expected:
        raise RuntimeError(f'Pacote com arquivos inesperados/ausentes: {found}. Refaça python scripts/prepare_hf.py')

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type='model', private=False, exist_ok=True)
    api.upload_folder(folder_path=str(package_dir), repo_id=args.repo_id, repo_type='model',
                      commit_message='Publish fraud detection model and model card')
    print(f'Modelo publicado: https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
