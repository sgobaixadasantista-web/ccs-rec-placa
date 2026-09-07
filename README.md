# CCS Reconhecimento de Placas

Protótipo para reconhecimento de placas veiculares brasileiras usando **YOLOv11 + OpenCV + EasyOCR**.

## Fluxo atual

1. Receber uma imagem do veículo.
2. YOLO localizar automaticamente a placa.
3. Recortar a região detectada.
4. Melhorar contraste e gerar múltiplos pré-processamentos.
5. Executar OCR com EasyOCR.
6. Normalizar e corrigir caracteres ambíguos conforme a posição.
7. Validar os formatos brasileiros:
   - Antigo: `ABC1234`
   - Mercosul: `ABC1D23`
8. Retornar placa, modelo, bbox e confianças.

A integração com Supabase será adicionada depois que o reconhecimento estiver confiável.

## Modelo YOLO padrão

O projeto usa por padrão o modelo público:

`felipedutrain/placa-br-yolov11`

Hugging Face:

`https://huggingface.co/felipedutrain/placa-br-yolov11`

O modelo localiza placas brasileiras. A leitura dos caracteres é feita pelo EasyOCR.

Arquivo usado automaticamente:

```text
https://huggingface.co/felipedutrain/placa-br-yolov11/resolve/main/best.pt
```

## OCR

O pipeline principal usa **EasyOCR**.

São testadas várias versões do mesmo recorte:

- imagem ampliada;
- escala de cinza;
- CLAHE;
- Otsu;
- limiarização adaptativa.

As leituras são classificadas pelos padrões brasileiros e o melhor candidato é retornado.

O `reconhecer_placa.py` antigo continua disponível como teste legado com Tesseract, por isso `pytesseract` permanece nas dependências por enquanto.

Na primeira execução, o EasyOCR pode baixar os pesos do modelo de reconhecimento automaticamente.

## Instalação

```bash
pip install -r requirements.txt
```

## Executar reconhecimento completo

```bash
python detector_yolo.py foto.jpg --salvar-recortes
```

Também é possível usar um modelo YOLO local:

```bash
python detector_yolo.py foto.jpg --modelo weights/license_plate.pt --salvar-recortes
```

## Dataset público Roboflow para placas brasileiras

Durante a validação, foi verificado que o projeto `placas-brasileiras/placas-0iqjy` do Roboflow é de **placas de trânsito**, não de placas veiculares. Ele não deve ser usado neste projeto.

Para placas veiculares brasileiras, o projeto usa como opção pública principal:

```text
workspace: akler
project: lpr-placas-brasileiras-2025-y4ydq
version: 8
```

Roboflow Universe:

```text
https://universe.roboflow.com/akler/lpr-placas-brasileiras-2025-y4ydq
```

Esse dataset possui classes `new` e `old`, correspondentes aos dois estilos de placas. Para o detector deste projeto, ambas são remapeadas para uma única classe:

```text
0 = license_plate
```

A distinção entre placa antiga e Mercosul continua sendo feita no OCR/validador.

### Baixar e preparar automaticamente

Configure sua API key do Roboflow:

Linux/macOS:

```bash
export ROBOFLOW_API_KEY="SUA_CHAVE"
```

Windows PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="SUA_CHAVE"
```

Depois execute:

```bash
python baixar_roboflow.py
```

O script baixa a versão padrão do dataset, copia os splits para o layout do projeto e converte todas as labels para `license_plate`.

Estrutura final:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

O arquivo `dataset.yaml` já está configurado para esses três splits.

## Benchmark com UFPR-ALPR

O projeto também possui o script:

```text
converter_ufpr_yolo.py
```

Ele converte as anotações locais do UFPR-ALPR para o formato YOLO usando uma única classe `license_plate`.

As imagens do UFPR-ALPR não devem ser adicionadas ou redistribuídas neste repositório; consulte os termos próprios do dataset.

## Regras de validação

### Placa antiga

```text
AAA0000
ABC1234
```

### Mercosul

```text
AAA0A00
ABC1D23
```

O sistema também tenta corrigir confusões comuns de OCR de acordo com a posição esperada, como `O/0`, `I/1`, `Q/0`, `G/6`, `B/8`, `S/5` e `Z/2`.

## Status

Já temos:

- detector YOLOv11 pré-treinado para placas brasileiras;
- EasyOCR como OCR principal;
- múltiplos pré-processamentos por placa;
- correção por máscara para placa antiga e Mercosul;
- retorno de `bbox`, confiança YOLO, confiança OCR e motor OCR;
- opção para salvar o recorte detectado;
- importador de dataset público do Roboflow;
- conversor UFPR-ALPR para benchmark local;
- `dataset.yaml` com train/val/test;
- estrutura pronta para fine-tuning;
- caminho aberto para integração com Supabase.

## Próximas etapas

- baixar o dataset Roboflow autorizado;
- validar visualmente uma amostra das anotações;
- testar YOLO + EasyOCR em lote;
- medir precisão do detector e do OCR separadamente;
- fazer fine-tuning se necessário;
- processar vídeo/câmera em tempo real;
- salvar eventos válidos no Supabase.
