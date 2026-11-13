# GLOSSARY — AdOps / Arbitragem de Anúncios

Vocabulário canônico do domínio. Use exatamente estes termos em issues, análises,
nomes de variáveis e comentários. Não derive sinônimos sem atualizar este arquivo.

---

## KPIs Mestres

### RPS — Revenue per Session
Receita gerada por uma sessão de usuário. KPI mestre da análise de arbitragem.
É a unidade de comparação direta com o CPC.

```
RPS = Receita total / Total de sessões
```

**Não confundir com:** RPM de pageview (divide por páginas, não por sessões).

---

### RPMS — Revenue per Mille Sessions
Versão em escala de campanha do RPS. Mesma métrica, base mil.

```
RPMS = RPS × 1000 = Receita / Sessões × 1000
```

**Uso:** relatórios agregados, comparação com benchmarks de publishers.
**Evitar:** EPMV (sinônimo externo, não usar internamente).

---

### Margem
Rentabilidade relativa da operação de arbitragem.

```
Margem (%) = (RPS - CPC) / CPC × 100
```

**Zona de risco:** margem < 20%.
**Positiva:** RPS > CPC. **Negativa:** RPS < CPC = prejuízo por sessão.

---

### ROI — Return on Investment
Retorno sobre o investimento total no período.

```
ROI (%) = (Receita - Custo) / Custo × 100
```

**Diferença de Margem:** ROI é sobre o total do período; Margem é por sessão.

---

## Lado da Compra (Buy-Side)

### CPC — Cost per Click
Custo pago por clique (= sessão) na fonte de tráfego (Taboola, Mgid, Outbrain, etc.).
É o custo unitário da arbitragem.

---

### EPC — Earnings per Click
Receita gerada por clique. Equivalente ao RPS quando 1 clique = 1 sessão.

```
EPC = RPS (na maioria dos cenários)
```

---

### Fonte de tráfego
Plataforma de compra de mídia. Exemplos canônicos: Taboola, Mgid, Outbrain,
Facebook Ads, Google UAC. Nunca generalizar como "DSP" sem especificar a plataforma.

---

## Lado da Venda (Sell-Side)

### Match Rate
Percentual de requisições de anúncio que retornam ao menos uma oferta de demanda.

```
Match Rate = Requisições com resposta / Total de requisições × 100
```

**Referência saudável:** > 85%.
**Causa principal de queda:** floor price bloqueando demanda no GAM.
**Não confundir com:** Delivery Rate (são etapas diferentes do funil).

---

### Delivery Rate
Percentual de impressões efetivamente entregues sobre as requisições que retornaram
demanda (matchadas).

```
Delivery Rate = Impressões entregues / Requisições matchadas × 100
```

**Referência saudável:** > 90%.
**Causa principal de queda:** problema de viewability ou frequência cap.

---

### Fill Rate
Percentual de requisições de anúncio que resultaram em impressão.

```
Fill Rate = Impressões / Requisições × 100
```

**Uso restrito:** apenas para formatos out-of-page (Interstitial, Offerwall).
**Não usar para:** Top, Anchor, Banner — nesses formatos Fill Rate não é diagnóstico relevante.

---

### Ad Request eCPM
Eficiência real de um slot de anúncio, calculada sobre as requisições (não as impressões).

```
Ad Request eCPM = Receita / Requisições × 1000
```

**Por que importa:** captura o impacto de baixo Match Rate no valor do slot.
Um eCPM de impressão alto com Ad Request eCPM baixo = muitas requisições sem retorno.

---

### imp/PV — Impressões por Pageview
Densidade de entrega de anúncios por página carregada.

```
imp/PV = Total de impressões / Total de pageviews
```

**Queda de imp/PV:** menos oportunidades de impressão por página — problema de layout,
viewability ou carregamento de script.

---

### RPM de pageview
Receita por mil pageviews. Métrica de publishers tradicionais.

```
RPM = Receita / Pageviews × 1000
```

**Limitação na arbitragem:** não reflete o custo real (que é por sessão, não por página).
Usar RPMS para comparação com CPC.

---

## Formatos de Anúncio

### Interstitial
Anúncio de tela cheia exibido entre navegações. Formato out-of-page.

**Diagnóstico específico:**
- Separar sempre Match Rate de Delivery Rate — causas distintas
- Match Rate baixo com Viewability estável → floor price bloqueando demanda no GAM
- Fill Rate relevante para este formato

---

### Offerwall
Produto de aquisição de usuário em formato de mural de ofertas. Formato out-of-page.

**Regras de diagnóstico:**
- Correlaciona com **novos usuários** (sess=0_2), não com volume total de sessões
- Queda com LFT alto = problema de mix de audiência, não de monetização
- Queda com LFT baixo = problema de demanda ou configuração

---

### Top / Anchor / Banner
Formatos in-page padrão (topo, rodapé fixo, retângulo de conteúdo).

**Regras:**
- Fill Rate **não é diagnóstico** para estes formatos — não citar
- Foco: imp/PV e Ad Request eCPM

---

### Rewarded / Native
Formatos de engajamento (vídeo recompensado, conteúdo patrocinado).

**Diagnóstico:** CTR e completion rate antes de RPMS.

---

## GDPR / Privacidade

### TCF — Transparency and Consent Framework
Protocolo do IAB para propagação de consentimento de usuário para compradores de mídia.

**Versão atual requerida:** TCF 2.2.
**Flag obrigatória no wrapper:**
```js
window['gtag_enable_tcf_support'] = true;
```
Ausência desta flag = compradores não recebem sinal de consentimento → menor demanda
em GEOs regulados mesmo com Match Rate aparentemente ok.

---

### CMP — Consent Management Platform
Plataforma que coleta e gerencia o consentimento do usuário. Deve ser carregada
**antes** do script de anúncio (GPT). Race condition = anúncios sem consent.

---

### Consent Rate
Percentual de usuários que aceitaram todos os propósitos de coleta de dados.

**Impacto:** usuários sem consentimento recebem demanda limitada (non-personalized ads).
GEOs com consent rate < 60% tendem a ter RPMS 40-60% menor que GEOs com > 80%.

---

### LGPD
Lei Geral de Proteção de Dados (Brasil). Não exige TCF, mas boas práticas recomendam
CMP configurado para usuários BR.

---

## Plataformas e Infraestrutura

### GAM — Google Ad Manager
Servidor de anúncios principal. Ponto central de configuração de floor prices,
ordens de demanda e regras de preço.

---

### Floor Price
Preço mínimo de CPM configurado no GAM abaixo do qual a demanda é bloqueada.
Principal causa de queda de Match Rate no Interstitial.

---

### Wrapper
Container JavaScript que gerencia o carregamento de tags de anúncio, header bidding
e configuração de TCF. Configurado via AdSeleto Hub.

---

## Análise e Diagnóstico

### Separação compra × venda
Primeiro passo obrigatório de qualquer diagnóstico de ROI. Determinar se o problema
está no custo (buy-side) ou na receita (sell-side) antes de qualquer análise profunda.

**Regra:**
- RPS estável + CPC subiu → problema de compra
- CPC estável + RPS caiu → problema de venda
- Ambos se moveram → isolar qual se moveu primeiro (causa) e qual é consequência

---

### Contrafactual
Pergunta de diagnóstico: "se a variável X tivesse ficado igual ao período bom,
o ROI estaria positivo?"

Usado no buy-side para isolar se o problema é 100% de custo ou se há também
componente de venda.

---

### Período de referência / Período de queda
Dois intervalos de tempo comparados no diagnóstico:
- **Período de referência:** janela com performance saudável (baseline)
- **Período de queda:** janela com performance degradada (a investigar)

---

### Human Approval Gate
Regra de governança do JCM: qualquer recomendação que implique mudança de orçamento,
floor price ou bid requer confirmação explícita do usuário antes de qualquer execução.
O JCM diagnostica e recomenda — nunca executa sem aprovação.
