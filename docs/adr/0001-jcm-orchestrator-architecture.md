# ADR-0001 — Arquitetura do Orquestrador JCM

**Status:** Aceito  
**Data:** 2026-11-13  
**Autor:** Lucas Tasso / AdSeleto

---

## Contexto

A análise de arbitragem de anúncios envolve duas dimensões independentes — compra de tráfego
(buy-side) e monetização (sell-side) — além de uma camada transversal de conformidade com
privacidade (GDPR/TCF). Diagnósticos manuais misturavam as duas dimensões com frequência,
gerando conclusões incorretas (ex: atribuir queda de ROI ao CPC quando o problema era Match
Rate do Interstitial).

Precisávamos de um agente capaz de:
1. Separar compra de venda como primeiro passo obrigatório
2. Especializar a análise por dimensão sem explodir o contexto do agente principal
3. Produzir um veredicto acionável com causa-raiz única e recomendações priorizadas
4. Nunca executar mudanças de orçamento ou floor price sem aprovação humana explícita

---

## Decisão

Adotar uma arquitetura de **orquestrador único com subagentes especializados**, comunicando-se
via **protocolo de mailbox em arquivo**.

### Agentes

```
jcm-orchestrator          ← ponto de entrada, intake, delegação
├── jcm-sell-side         ← Match Rate, Delivery Rate, RPMS, imp/PV, Ad Request eCPM
├── jcm-buy-side          ← CPC, EPC, Margem, qualidade de tráfego, contrafactual
├── jcm-gdpr-auditor      ← TCF, CMP, consent rate, impacto por GEO regulado
└── jcm-verdict-writer    ← consolida os 3 outboxes, escreve diagnóstico final
```

### Protocolo de mailbox

Cada subagente escreve seu resultado em:
```
.claude/mailboxes/<agente>/outbox/<arquivo>.md
```

O orquestrador passa apenas o **caminho do arquivo** para o verdict-writer — nunca carrega
o conteúdo completo no seu próprio contexto. Isso mantém o contexto do orchestrador limpo
independentemente do volume de dados analisados.

### Modelos por agente

| Agente | Modelo | Justificativa |
|---|---|---|
| jcm-orchestrator | claude-opus-4-5 | Raciocínio estratégico, coordenação |
| jcm-sell-side | claude-sonnet-4-5 | Análise técnica densa, custo balanceado |
| jcm-buy-side | claude-sonnet-4-5 | Análise técnica densa, custo balanceado |
| jcm-gdpr-auditor | claude-haiku-4-5 | Checklist estruturado, baixo custo |
| jcm-verdict-writer | claude-sonnet-4-5 | Síntese e redação, custo balanceado |

### Fluxo de execução

```
1. Intake (orchestrator)
   └── Coleta: site, período, fonte, KPI afetado

2. Separação compra × venda (orchestrator)
   └── RPS estável + CPC subiu → buy-side
   └── CPC estável + RPS caiu → sell-side
   └── Ambos → isolar causa e consequência

3. Delegação paralela
   ├── sell-side ou buy-side (conforme diagnóstico)
   └── gdpr-auditor (sempre, em paralelo, não bloqueia)

4. Consolidação (verdict-writer)
   └── Lê outboxes → escreve veredicto final no chat
```

### Human Approval Gate

Qualquer recomendação que implique mudança de orçamento, floor price ou bid é sinalizada
com ⚠️ e requer confirmação explícita do usuário. O JCM diagnostica e recomenda —
nunca executa sem aprovação.

---

## Alternativas consideradas

### Agente único com contexto completo
Um único agente recebendo todos os dados de compra, venda e privacidade.

**Rejeitado porque:**
- Contexto cresce linearmente com o volume de dados → respostas mais lentas e caras
- Maior risco de misturar raciocínio de compra com venda no mesmo prompt
- Difícil especializar o comportamento por dimensão sem instruções muito longas

### Multi-agente com API calls diretas (sem mailbox)
Subagentes retornando resultados diretamente ao orchestrador via tool call.

**Rejeitado porque:**
- O conteúdo completo da análise entra no contexto do orchestrador
- Para análises densas (séries temporais, tabelas de dados) isso é proibitivo
- Mailbox desacopla o volume de dados do tamanho do contexto

### Pipeline sequencial obrigatório (sell → buy → gdpr)
Executar os três subagentes em sequência, um bloqueando o próximo.

**Rejeitado porque:**
- gdpr-auditor é independente dos outros dois e pode rodar em paralelo
- Latência desnecessária quando o problema é claramente de um único lado

---

## Consequências

**Positivas:**
- Contexto do orchestrador permanece pequeno independentemente do volume de dados
- Especialização real por dimensão: cada subagente tem protocolo e KPIs específicos
- gdpr-auditor em paralelo não adiciona latência ao caminho crítico
- Human Approval Gate como regra arquitetural, não como lembrete de prompt

**Negativas / trade-offs:**
- Dependência de sistema de arquivos para comunicação entre agentes (mailbox)
- Verdict-writer precisa de acesso de leitura aos outboxes de todos os outros agentes
- Custo total maior que agente único (5 chamadas de modelo vs 1), compensado pela
  especialização e pelo uso de Haiku para o auditor de baixa complexidade

---

## Referências

- `GLOSSARY.md` — vocabulário canônico do domínio
- `.claude/agents/jcm-orchestrator.md` — protocolo completo do orchestrador
- `.claude/agents/jcm-sell-side.md` — protocolo de análise de venda
- `.claude/agents/jcm-buy-side.md` — protocolo de análise de compra
- `.claude/agents/jcm-gdpr-auditor.md` — checklist de conformidade
- `.claude/agents/jcm-verdict-writer.md` — estrutura do veredicto final
