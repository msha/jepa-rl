export async function api(path: string, payload?: Record<string, unknown>): Promise<Record<string, unknown>> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  })
  const data = await res.json()
  if (!res.ok || data.ok === false) throw new Error(data.error || res.statusText)
  return data as Record<string, unknown>
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path)
  return res.json() as Promise<T>
}
