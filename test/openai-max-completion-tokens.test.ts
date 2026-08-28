import { describe, expect, it, afterEach, vi } from "vitest";
import { OpenAIProvider } from "../src/providers/openai.js";

/**
 * gpt-5 family and o-series deployments reject `max_tokens`:
 *
 *   400 {"error":{"message":"Unsupported parameter: 'max_tokens' is not
 *   supported with this model. Use 'max_completion_tokens' instead.", ...}}
 *
 * The spelling an endpoint accepts cannot be derived from the model string —
 * Azure deployment names are user-chosen — so the provider learns it from the
 * first rejection and keeps it for the rest of the process.
 */

const UNSUPPORTED_MAX_TOKENS = JSON.stringify({
  error: {
    message:
      "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
    type: "invalid_request_error",
    param: "max_tokens",
    code: "unsupported_parameter",
  },
});

function ok(content: string): Response {
  return new Response(JSON.stringify({ choices: [{ message: { content } }] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function bodyOf(call: unknown[]): Record<string, unknown> {
  const init = call[1] as RequestInit;
  return JSON.parse(init.body as string) as Record<string, unknown>;
}

describe("OpenAIProvider — max_tokens vs max_completion_tokens", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends max_tokens by default", async () => {
    const fetchMock = vi.fn().mockResolvedValue(ok("<observation/>"));
    vi.stubGlobal("fetch", fetchMock);

    const provider = new OpenAIProvider("k", "gpt-4o-mini", 800, "https://api.example.com");
    await provider.compress("sys", "user");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const body = bodyOf(fetchMock.mock.calls[0]!);
    expect(body["max_tokens"]).toBe(800);
    expect(body["max_completion_tokens"]).toBeUndefined();
  });

  it("retries with max_completion_tokens when the API rejects max_tokens", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(UNSUPPORTED_MAX_TOKENS, { status: 400 }))
      .mockResolvedValueOnce(ok("<observation/>"));
    vi.stubGlobal("fetch", fetchMock);

    const provider = new OpenAIProvider("k", "gpt-5.4-mini", 800, "https://api.example.com");
    const result = await provider.compress("sys", "user");

    expect(result).toBe("<observation/>");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(bodyOf(fetchMock.mock.calls[0]!)["max_tokens"]).toBe(800);
    const retry = bodyOf(fetchMock.mock.calls[1]!);
    expect(retry["max_completion_tokens"]).toBe(800);
    expect(retry["max_tokens"]).toBeUndefined();
  });

  it("keeps the learned spelling so later calls cost no extra round trip", async () => {
    const fetchMock = vi
      .fn()
      // a Response body can only be read once, so hand out a fresh one per call
      .mockImplementationOnce(async () => new Response(UNSUPPORTED_MAX_TOKENS, { status: 400 }))
      .mockImplementation(async () => ok("<observation/>"));
    vi.stubGlobal("fetch", fetchMock);

    const provider = new OpenAIProvider("k", "gpt-5.4-mini", 800, "https://api.example.com");
    await provider.compress("sys", "first");
    await provider.compress("sys", "second");

    // 2 for the first call (reject + retry), 1 for the second
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(bodyOf(fetchMock.mock.calls[2]!)["max_completion_tokens"]).toBe(800);
  });

  it("does not retry on unrelated 400s", async () => {
    const fetchMock = vi.fn().mockImplementation(
      async () =>
        new Response(JSON.stringify({ error: { message: "context_length_exceeded" } }), {
          status: 400,
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const provider = new OpenAIProvider("k", "gpt-4o-mini", 800, "https://api.example.com");
    await expect(provider.compress("sys", "user")).rejects.toThrow(/context_length_exceeded/);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
