import cors from "cors";
import express from "express";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OpenAI} from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter";
import * as dotenv from 'dotenv'
dotenv.config()
 const loader = new PDFLoader("CC notes 1.pdf"); //you can change this to any PDF file of your choice.

 const docs = await loader.load();
// console.log('docs loaded',docs)
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 20,
  }); 
const splitteddocs = await splitter.splitDocuments(docs)
//  console.log(splitteddocs,"chunks")
// embedding chunks
const embeddings = new OpenAIEmbeddings();
//...................................................
  const vectorStore = await FaissStore.fromDocuments(
    splitteddocs,embeddings
  );
// console.log(vectorStore)
  const vectorStoreRetriever = vectorStore.asRetriever();
  const model = new OpenAI({
    modelName:"gpt-3.5-turbo"
  });
  const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);
// console.log(chain,"chain pk")
var app=express()
app.use(express.json())
app.use(cors())
app.post("/ask",async(req,resp)=>{
  var question=req.body.q
  console.log(req.body.q)
  var answer = await chain.call({
    query: question,
  }); 
  resp.send({"ans":answer})
  console.log(answer,"ans")
})

// //.....................................................................

app.listen(9000)