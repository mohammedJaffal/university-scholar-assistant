package com.fsbmchatbot.fsbmchatbotbackend.controller;

import com.fsbmchatbot.fsbmchatbotbackend.dto.DocumentResponse;
import com.fsbmchatbot.fsbmchatbotbackend.dto.MessageResponse;
import com.fsbmchatbot.fsbmchatbotbackend.security.UserDetailsImpl; // Ensure this import
import com.fsbmchatbot.fsbmchatbotbackend.service.DocumentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@CrossOrigin(origins = "*", maxAge = 3600)
@RestController
@RequestMapping("/api/documents")
public class DocumentController {

    @Autowired
    private DocumentService documentService;

    @PostMapping
    @PreAuthorize("hasRole('PROFESSOR')")
    public ResponseEntity<DocumentResponse> uploadDocument(@RequestParam("file") MultipartFile file,
                                                           @RequestParam("moduleId") Long moduleId,
                                                           @AuthenticationPrincipal UserDetailsImpl currentUser) {
        if (currentUser == null) { // Should be handled by security, but defensive check
            return new ResponseEntity<>(HttpStatus.UNAUTHORIZED);
        }
        DocumentResponse documentResponse = documentService.storeDocument(file, moduleId, currentUser.getId());
        return new ResponseEntity<>(documentResponse, HttpStatus.CREATED);
    }

    @GetMapping("/my-documents")
    @PreAuthorize("hasRole('PROFESSOR')")
    public ResponseEntity<List<DocumentResponse>> getMyDocuments(@AuthenticationPrincipal UserDetailsImpl currentUser) {
        if (currentUser == null) {
            return new ResponseEntity<>(HttpStatus.UNAUTHORIZED);
        }
        List<DocumentResponse> documents = documentService.getDocumentsByUploader(currentUser.getId());
        return ResponseEntity.ok(documents);
    }

    @GetMapping("/all")
    @PreAuthorize("hasRole('ADMINISTRATOR')") // Admin can see all documents
    public ResponseEntity<List<DocumentResponse>> getAllDocuments() {
        List<DocumentResponse> documents = documentService.getAllDocuments();
        return ResponseEntity.ok(documents);
    }

    @GetMapping("/{documentId}")
    @PreAuthorize("isAuthenticated()") // Any authenticated user can try to get details
    public ResponseEntity<DocumentResponse> getDocumentById(@PathVariable Long documentId) {
        DocumentResponse document = documentService.getDocumentResponseById(documentId);
        return ResponseEntity.ok(document);
    }

    @PutMapping("/{documentId}")
    @PreAuthorize("hasRole('PROFESSOR')")
    public ResponseEntity<DocumentResponse> updateDocumentModule(@PathVariable Long documentId,
                                                                 @RequestParam("moduleId") Long newModuleId,
                                                                 @AuthenticationPrincipal UserDetailsImpl currentUser) {
        if (currentUser == null) {
            return new ResponseEntity<>(HttpStatus.UNAUTHORIZED);
        }
        DocumentResponse updatedDocument = documentService.updateDocumentModule(documentId, newModuleId, currentUser.getId());
        return ResponseEntity.ok(updatedDocument);
    }

    @DeleteMapping("/{documentId}")
    @PreAuthorize("hasRole('PROFESSOR')")
    public ResponseEntity<MessageResponse> deleteDocument(@PathVariable Long documentId,
                                                          @AuthenticationPrincipal UserDetailsImpl currentUser) {
        if (currentUser == null) {
            return new ResponseEntity<>(HttpStatus.UNAUTHORIZED);
        }
        documentService.deleteDocument(documentId, currentUser.getId());
        return ResponseEntity.ok(new MessageResponse("Document deleted successfully."));
    }

    @GetMapping("/download/{documentId}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<Resource> downloadDocument(@PathVariable Long documentId) {
        // Assuming DocumentService has these methods
        Resource resource = documentService.loadDocumentAsResource(documentId);
        String originalFilename = documentService.getOriginalDocumentName(documentId);

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_OCTET_STREAM) // Generic binary stream
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + originalFilename + "\"")
                .body(resource);
    }
}