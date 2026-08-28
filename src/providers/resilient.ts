import type { MemoryProvider, CircuitBreakerState } from "../types.js";
import { CircuitBreaker } from "./circuit-breaker.js";

/**
 * A rejection that means "this particular payload is not acceptable" rather
 * than "the provider is unhealthy".
 *
 * Azure OpenAI content filters are the motivating case. Prompt Shields flags
 * tool output that merely *looks* like a jailbreak — a README describing
 * prompt injection, a security test fixture, an error log quoting user input:
 *
 *   400 {"error":{"code":"content_filter", ...
 *        "innererror":{"code":"ResponsibleAIPolicyViolation",
 *        "content_filter_result":{"jailbreak":{"detected":true,"filtered":true}}}}}
 *
 * Counting those as provider failures means three filtered observations inside
 * the failure window trip the breaker, and every *other* compression then
 * fails with `circuit_breaker_open` for the recovery timeout. One awkward file
 * costs a batch of unrelated observations. The provider is answering fine, so
 * leave the breaker closed and let just that one call fail.
 */
export function isPayloadRejection(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err);
  return /content_filter|ResponsibleAIPolicyViolation/.test(message);
}

export class ResilientProvider implements MemoryProvider {
  private breaker = new CircuitBreaker();
  name: string;

  constructor(private inner: MemoryProvider) {
    this.name = `resilient(${inner.name})`;
  }

  private async call(fn: () => Promise<string>): Promise<string> {
    if (!this.breaker.isAllowed) {
      throw new Error("circuit_breaker_open");
    }
    try {
      const result = await fn();
      this.breaker.recordSuccess();
      return result;
    } catch (err) {
      if (!isPayloadRejection(err)) {
        this.breaker.recordFailure();
      }
      throw err;
    }
  }

  async compress(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call(() => this.inner.compress(systemPrompt, userPrompt));
  }

  async summarize(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call(() => this.inner.summarize(systemPrompt, userPrompt));
  }

  get circuitState(): CircuitBreakerState {
    return this.breaker.getState();
  }
}
