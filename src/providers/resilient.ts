import type { MemoryProvider, CircuitBreakerState } from "../types.js";
import { CircuitBreaker } from "./circuit-breaker.js";

// Azure Prompt Shields rejects ordinary material — a SECURITY.md describing
// prompt injection is enough — so these arrive often enough to trip the
// breaker and take unrelated compressions down with them. See #1276.
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
