# RAG Evaluation Report — ✅ PASS

Provider: `mock` · Retriever: `bm25` · Cases: 25 · Chunks: 12 · Cases fully passed: 22/25

| Metric | Score | Min | Status |
|---|---|---|---|
| retrieval_hit_rate | 1.00 | 0.90 | ✅ |
| answer_correctness | 0.90 | 0.80 | ✅ |
| groundedness | 1.00 | 0.85 | ✅ |
| citation_accuracy | 1.00 | 0.85 | ✅ |
| refusal_accuracy | 0.92 | 0.80 | ✅ |

## Cases

| ID | Pass | Answer | Notes |
|---|---|---|---|
| leave-annual | ✅ | Full-time employees receive 24 days of paid annual leave per calendar year. [1] |  |
| leave-carryover | ✅ | Up to 5 unused leave days can be carried over into the next year. [1] |  |
| sick-certificate | ✅ | A medical certificate is required for sick leave longer than 3 consecutive days. [1] |  |
| parental-primary | ✅ | Parental leave is 26 weeks for the primary caregiver and 4 weeks for the secondary caregiver, both fully paid. [1] |  |
| remote-days | ✅ | Employees may work remotely up to 3 days per week. [1] |  |
| office-days | ✅ | Tuesdays and Thursdays are in-office collaboration days for all teams. [1] |  |
| expense-deadline | ✅ | Expense claims must be submitted within 30 days of the purchase through the Expensify portal. [1] |  |
| expense-preapproval | ✅ | Any single expense above 10,000 INR requires manager pre-approval. [1] |  |
| password-length | ❌ | I don't know based on the provided documents. | missing facts: ['14 characters']; refused an in-scope question |
| lost-laptop | ✅ | Lost or stolen laptops must be reported to security@brightleaf.example within 24 hours. [1] |  |
| core-hours | ✅ | Core working hours are 11:00 to 16:00 IST, when everyone is expected to be available for meetings. [1] |  |
| price-team | ✅ | The Team plan costs 4,500 INR per user per month and includes unlimited dashboards and scheduled email reports. [1] The Starter plan costs 2… |  |
| trial | ❌ | No credit card is required to start the trial. [1] | missing facts: ['14-day'] |
| sso | ✅ | The Enterprise plan has custom pricing and adds single sign-on (SSO), audit logs and a 99.9% uptime SLA. [1] |  |
| data-sources | ✅ | Brightleaf Insights supports PostgreSQL, MySQL, Snowflake and BigQuery as data sources. [1] |  |
| retention | ✅ | Query history is kept for 90 days on Starter and Team plans and for 1 year on Enterprise. [1] |  |
| browsers | ✅ | End-to-end UI tests are automated with Playwright and run on Chromium, Firefox and WebKit. [1] |  |
| exit-critical | ✅ | The release can ship when there are zero open critical defects and no more than 3 open major defects. [1] |  |
| p95 | ✅ | The p95 dashboard load time must be under 2 seconds at 500 concurrent users. [1] |  |
| regression-pass | ✅ | Automated regression pass rate must be at least 98 percent. [1] |  |
| oos-ceo | ❌ | Brightleaf Analytics is a fictional 120-person data company used as sample content for this project. [1] | answered an out-of-scope question |
| oos-stock | ✅ | I don't know based on the provided documents. |  |
| oos-pizza | ✅ | I don't know based on the provided documents. |  |
| oos-gym | ✅ | I don't know based on the provided documents. |  |
| oos-weather | ✅ | I don't know based on the provided documents. |  |
