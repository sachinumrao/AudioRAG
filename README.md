# AudioRAG
Build RAG based Question-Answering bot for youtube videos using Whisper and LLMs.

## Components
- [x] Get audio data from url
- [x] Transcribe audio to english text
    - [x] Huggingface ASR pipeline
    - [x] openai whisper support
    - [ ] faster-whisper support
    - [ ] whisper-cpp support
- [ ] Audio splitter based on pauses (optional)
- [ ] Create chunks of transcripts using sliding window method
- [ ] Create embeddings of chunks with encoder model
- [ ] Create index of chunks
    - [ ] Support for voyager as Index
    - [ ] Support for ElasticSearch as Index (optional)
- [ ] CRUD on Index
- [ ] LLM response generation
- [ ] Build UI for interactive demo