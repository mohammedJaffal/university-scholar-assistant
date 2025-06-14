package com.fsbmchatbot.fsbmchatbotbackend.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor; // Optional: if you want a constructor with all args
/**
 * Represents the expected JSON response structure from the external Python RAG Chatbot API.
 * Field names in this DTO MUST match the field names in the JSON sent by the Python API.
 */
@Data // Lombok: Generates getters, setters, toString, equals, hashCode
@NoArgsConstructor // Lombok: Generates a no-argument constructor
@AllArgsConstructor // Optional: Lombok: Generates a constructor with all fields as arguments
public class ChatQueryResponse {

    /**
     * The main textual response/answer from the chatbot.
     * Ensure the JSON key from Python API matches this field name (e.g., "response", "answer", "message").
     * If your Python API uses "answer", change this field to `private String answer;`
     */
    private String response;

    /**
     * Optional: If your Python RAG API returns a list of source document names or snippets.
     * Ensure the JSON key from Python API matches this field name (e.g., "sources", "references").
     */
    // private List<String> sources;

    /**
     * Optional: If your Python RAG API returns any other relevant data,
     * add corresponding fields here.
     * Example:
     * private double confidenceScore;
     * private String intentDetected;
     */

    // Example constructor if you don't use @AllArgsConstructor and want a specific one
    // public ChatQueryResponse(String response) {
    //     this.response = response;
    // }
}