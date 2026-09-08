import csv
import sys
from pathlib import Path


def distancia_hamming(a: str, b: str):
    if len(a) != len(b):
        return None
    return sum(x != y for x, y in zip(a, b))


def main():
    if len(sys.argv) != 3:
        print('Uso: python avaliar_benchmark.py benchmark_placas.csv resultado_lote/resumo.csv')
        raise SystemExit(2)

    gabarito_path = Path(sys.argv[1])
    resultado_path = Path(sys.argv[2])

    with gabarito_path.open(encoding='utf-8', newline='') as f:
        gabarito = {r['arquivo']: r['placa_real'].strip().upper() for r in csv.DictReader(f)}

    with resultado_path.open(encoding='utf-8', newline='') as f:
        resultados = {r['arquivo']: r for r in csv.DictReader(f)}

    linhas = []
    exatos = 0
    total = len(gabarito)
    chars_corretos = 0
    chars_total = 0

    for arquivo, placa_real in gabarito.items():
        r = resultados.get(arquivo, {})
        placa_lida = (r.get('placa') or '').strip().upper()
        exato = placa_lida == placa_real
        if exato:
            exatos += 1

        corretos = sum(a == b for a, b in zip(placa_real, placa_lida))
        chars_corretos += corretos
        chars_total += len(placa_real)
        dist = distancia_hamming(placa_real, placa_lida)

        linhas.append({
            'arquivo': arquivo,
            'placa_real': placa_real,
            'placa_lida': placa_lida,
            'acerto_exato': exato,
            'caracteres_corretos': corretos,
            'caracteres_total': len(placa_real),
            'distancia_hamming': '' if dist is None else dist,
            'confianca_yolo': r.get('confianca_yolo', ''),
            'confianca_ocr': r.get('confianca_ocr', ''),
            'deteccoes': r.get('deteccoes', ''),
        })

    saida = resultado_path.parent / 'benchmark.csv'
    with saida.open('w', encoding='utf-8', newline='') as f:
        campos = list(linhas[0].keys()) if linhas else []
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)

    taxa_exata = (exatos / total * 100) if total else 0.0
    taxa_chars = (chars_corretos / chars_total * 100) if chars_total else 0.0

    resumo = resultado_path.parent / 'benchmark_resumo.txt'
    texto = (
        f'total={total}\n'
        f'acertos_exatos={exatos}\n'
        f'taxa_acerto_exato={taxa_exata:.2f}%\n'
        f'caracteres_corretos={chars_corretos}\n'
        f'caracteres_total={chars_total}\n'
        f'taxa_acerto_caracteres={taxa_chars:.2f}%\n'
    )
    resumo.write_text(texto, encoding='utf-8')

    print('===== BENCHMARK =====')
    print(texto, end='')
    print('\n===== DETALHES =====')
    for l in linhas:
        print(f"{l['arquivo']}: real={l['placa_real']} lida={l['placa_lida']} exato={l['acerto_exato']} chars={l['caracteres_corretos']}/{l['caracteres_total']} hamming={l['distancia_hamming']}")


if __name__ == '__main__':
    main()
