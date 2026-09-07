# UFPR-ALPR no CCS Reconhecimento de Placas

O UFPR-ALPR e um dataset academico da Universidade Federal do Parana com imagens reais de veiculos e anotacoes de placa.

## Importante sobre a licenca

O dataset nao deve ser incluido ou redistribuido neste repositorio.

A pagina oficial informa que o uso e restrito a pesquisa academica e nao comercial, e que redistribuicao, modificacao ou uso comercial exigem permissao do VRI Lab.

Por isso, este projeto inclui somente o **conversor de anotacoes**. O usuario deve obter o dataset diretamente com os autores e manter as imagens localmente.

Pagina oficial:

https://web.inf.ufpr.br/vri/databases/ufpr-alpr/

Licenca:

https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/

## Objetivo

Converter a posicao da placa informada nos arquivos TXT do UFPR-ALPR para o formato YOLO:

```text
0 x_center y_center width height
```

A classe `0` representa:

```text
license_plate
```

A distincao entre placa antiga e Mercosul continua sendo feita pelo OCR e pelas regras de validacao do projeto.

## Uso

Depois de obter e extrair o dataset localmente:

```bash
python converter_ufpr_yolo.py --origem /caminho/UFPR-ALPR --destino dataset/ufpr_yolo
```

O conversor tenta preservar os splits oficiais quando encontrar pastas como:

```text
train/
validation/
test/
```

ou equivalentes.

## Estrutura gerada

```text
dataset/ufpr_yolo/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  dataset_ufpr.yaml
```

## Formato da bbox

Por padrao, o script interpreta os quatro numeros da anotacao de posicao da placa como:

```text
x y width height
```

Se a versao do conjunto de anotacoes que voce recebeu estiver em coordenadas de cantos:

```text
x1 y1 x2 y2
```

execute:

```bash
python converter_ufpr_yolo.py --origem /caminho/UFPR-ALPR --bbox-format xyxy
```

## Validacao recomendada

Antes de treinar, abra algumas imagens e confira visualmente as labels convertidas. Isso evita treinar com caixas incorretas caso a estrutura da versao recebida seja diferente.

Depois da conversao, o dataset pode ser usado para benchmark/fine-tuning do detector, respeitando integralmente os termos de uso do UFPR-ALPR.
