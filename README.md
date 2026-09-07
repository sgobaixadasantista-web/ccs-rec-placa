# CCS Reconhecimento de Placas

Protótipo inicial para reconhecimento de placas veiculares brasileiras usando OpenCV + Tesseract OCR.

## Objetivo

Fluxo inicial:

1. Receber uma imagem do veículo.
2. Recortar ou localizar a região da placa.
3. Melhorar contraste e binarizar a imagem.
4. Executar OCR.
5. Normalizar o texto.
6. Validar os formatos brasileiros:
   - Antigo: `ABC1234`
   - Mercosul: `ABC1D23`

A integração com Supabase será adicionada depois que o reconhecimento estiver confiável.

## Instalação

```bash
pip install -r requirements.txt
```

Também é necessário instalar o Tesseract OCR no sistema operacional.

### Windows

Instale o Tesseract OCR e, se necessário, configure no Python:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

## Executar

```bash
python reconhecer_placa.py caminho/para/foto.jpg
```

Exemplo de saída:

```python
{
    'placa': 'ABC1D23',
    'valida': True,
    'ocr_bruto': 'ABC1D23'
}
```

## Status atual

Em uma imagem de teste real, o Tesseract conseguiu chegar próximo da leitura correta, mas ainda apresentou confusão entre caracteres visualmente semelhantes, como `Q/D/G` e `0/O`.

Próximas melhorias planejadas:

- detector automático da região da placa;
- múltiplos pré-processamentos e votação entre OCRs;
- correção contextual pelo padrão Mercosul;
- score de confiança;
- YOLO para detecção da placa;
- integração posterior com Supabase.
