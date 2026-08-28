import { describe, expect, it, vi } from "vitest";
import { ResilientProvider, isPayloadRejection } from "../src/providers/resilient.js";
import type { MemoryProvider } from "../src/types.js";

/**
 * Azure Prompt Shields rejects tool output that looks like a jailbreak — which
 * includes benign material such as a SECURITY.md explaining prompt injection.
 * Those rejections must not open the circuit breaker, or one awkward file
 * takes every unrelated compression down with it for the recovery timeout.
 */

const CONTENT_FILTER_ERROR = new Error(
  'OpenAI API error (400): {"error":{"message":"The response was filtered due to ' +
    'the prompt triggering Azure OpenAI\'s content management policy.",' +
    '"code":"content_filter","status":400,"innererror":{"code":"ResponsibleAIPolicyViolation",' +
    '"content_filter_result":{"jailbreak":{"detected":true,"filtered":true}}}}}',
);

function providerThatFails(err: Error, okAfter = Infinity): MemoryProvider {
  let calls = 0;
  return {
    name: "stub",
    async compress() {
      calls += 1;
      if (calls > okAfter) return "<observation/>";
      throw err;
    },
    async summarize() {
      return "<summary/>";
    },
  } as MemoryProvider;
}

describe("isPayloadRejection", () => {
  it("recognises Azure content filter rejections", () => {
    expect(isPayloadRejection(CONTENT_FILTER_ERROR)).toBe(true);
    expect(isPayloadRejection(new Error("ResponsibleAIPolicyViolation"))).toBe(true);
  });

  it("treats genuine provider trouble as a failure", () => {
    expect(isPayloadRejection(new Error("OpenAI API error (503): upstream down"))).toBe(false);
    expect(isPayloadRejection(new Error("request timed out after 60000ms"))).toBe(false);
    expect(isPayloadRejection("socket hang up")).toBe(false);
  });
});

describe("ResilientProvider — content filter vs circuit breaker", () => {
  it("keeps the breaker closed across repeated filter rejections", async () => {
    const provider = new ResilientProvider(providerThatFails(CONTENT_FILTER_ERROR));

    for (let i = 0; i < 5; i++) {
      await expect(provider.compress("sys", "bad")).rejects.toThrow(/content_filter/);
    }

    expect(provider.circuitState.state).toBe("closed");
  });

  it("lets an unrelated call through after filtered ones", async () => {
    // fails with content_filter three times, succeeds from the fourth call on
    const provider = new ResilientProvider(providerThatFails(CONTENT_FILTER_ERROR, 3));

    for (let i = 0; i < 3; i++) {
      await expect(provider.compress("sys", "bad")).rejects.toThrow(/content_filter/);
    }

    // Before this change the fourth call threw circuit_breaker_open instead.
    await expect(provider.compress("sys", "harmless")).resolves.toBe("<observation/>");
  });

  it("still opens the breaker for real provider failures", async () => {
    const provider = new ResilientProvider(
      providerThatFails(new Error("OpenAI API error (503): upstream down")),
    );

    for (let i = 0; i < 3; i++) {
      await expect(provider.compress("sys", "x")).rejects.toThrow(/503/);
    }

    expect(provider.circuitState.state).toBe("open");
    await expect(provider.compress("sys", "x")).rejects.toThrow("circuit_breaker_open");
  });
});
