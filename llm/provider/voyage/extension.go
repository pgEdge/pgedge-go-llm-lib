//-------------------------------------------------------------------------
//
// pgEdge Go LLM Library
//
// Copyright (c) 2025 - 2026, pgEdge, Inc.
// This software is released under The PostgreSQL License
//
//-------------------------------------------------------------------------

// Package voyage implements the llm.Client interface for Voyage AI.
// Voyage offers text embeddings, multimodal embeddings, and rerankers;
// it does not offer chat completions, so Chat and ChatStream return
// llm.ErrNotSupported.
//
// Per-call options specific to Voyage (input_type, output_dimension,
// truncation, output_dtype, return_documents) are passed via the
// Extension type as a ProviderExtension.
package voyage

// InputType is Voyage's hint for whether an embedding text is a search
// query or a stored document. It affects retrieval quality and is the
// most-impactful Voyage-specific tuning knob.
type InputType string

// InputType values accepted by Voyage's embedding endpoints.
const (
	InputTypeQuery    InputType = "query"
	InputTypeDocument InputType = "document"
)

// OutputDtype is the numeric encoding for embedding vector components.
type OutputDtype string

// OutputDtype values accepted by Voyage's embedding endpoints.
const (
	OutputDtypeFloat   OutputDtype = "float"
	OutputDtypeInt8    OutputDtype = "int8"
	OutputDtypeUint8   OutputDtype = "uint8"
	OutputDtypeBinary  OutputDtype = "binary"
	OutputDtypeUbinary OutputDtype = "ubinary"
)

// Extension is a Voyage-specific per-call extension attached to
// MultimodalEmbedRequest.Extensions or RerankRequest.Extensions.
// Providers other than Voyage ignore it.
type Extension struct {
	InputType       InputType
	OutputDimension int   // 256 / 512 / 1024 / 2048, model-dependent; 0 = provider default
	Truncation      *bool // nil = provider default
	OutputDtype     OutputDtype
	ReturnDocuments *bool // rerank only; nil = provider default
}

// ProviderName returns "voyage" so llm.FindExtension can locate this
// extension in a request's Extensions slice.
func (Extension) ProviderName() string { return "voyage" }
