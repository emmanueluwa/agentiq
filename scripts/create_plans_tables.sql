CREATE TABLE IF NOT EXISTS plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
    source_message_id UUID NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    outcome TEXT NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_plans_conversation_id
    ON plans(conversation_id);

CREATE TABLE IF NOT EXISTS plan_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id UUID NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    details TEXT NOT NULL DEFAULT '',
    status VARCHAR(32) NOT NULL DEFAULT 'todo',
    position INTEGER NOT NULL,
    due_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_plan_items_plan_id
    ON plan_items(plan_id);

CREATE INDEX IF NOT EXISTS idx_plan_items_status
    ON plan_items(status);
