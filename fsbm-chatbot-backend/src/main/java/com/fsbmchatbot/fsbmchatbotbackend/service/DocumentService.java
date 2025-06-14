package com.fsbmchatbot.fsbmchatbotbackend.service;

import com.fsbmchatbot.fsbmchatbotbackend.dto.DocumentResponse;
import com.fsbmchatbot.fsbmchatbotbackend.exception.AppException;
import com.fsbmchatbot.fsbmchatbotbackend.exception.BadRequestException;
import com.fsbmchatbot.fsbmchatbotbackend.exception.FileStorageException;
import com.fsbmchatbot.fsbmchatbotbackend.exception.ResourceNotFoundException;
import com.fsbmchatbot.fsbmchatbotbackend.model.Document;
import com.fsbmchatbot.fsbmchatbotbackend.model.Module;
import com.fsbmchatbot.fsbmchatbotbackend.model.User;
import com.fsbmchatbot.fsbmchatbotbackend.repository.DocumentRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;


import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class DocumentService {

    private static final Logger logger = LoggerFactory.getLogger(DocumentService.class);

    private final Path fileStorageLocation;
    private final DocumentRepository documentRepository;
    private final UserService userService; // To get User entity
    private final ModuleService moduleService; // To get Module entity

    // Optional: RAGPipelineService for triggering indexing/de-indexing
    // @Autowired
    // private RAGPipelineService ragPipelineService;

    @Autowired
    public DocumentService(@Value("${file.upload-dir}") String uploadDir,
                           DocumentRepository documentRepository,
                           UserService userService,
                           ModuleService moduleService) {
        this.fileStorageLocation = Paths.get(uploadDir).toAbsolutePath().normalize();
        this.documentRepository = documentRepository;
        this.userService = userService;
        this.moduleService = moduleService;

        try {
            Files.createDirectories(this.fileStorageLocation);
            logger.info("Created file storage directory at: {}", this.fileStorageLocation);
        } catch (Exception ex) {
            throw new FileStorageException("Could not create the directory where the uploaded files will be stored.", ex);
        }
    }

    // Mapper method (can be in a separate Mapper class)
    private DocumentResponse convertToDocumentResponse(Document document) {
        String downloadUrl = ServletUriComponentsBuilder.fromCurrentContextPath()
                .path("/api/documents/download/")
                .path(String.valueOf(document.getId()))
                .toUriString();

        return new DocumentResponse(
                document.getId(),
                document.getOriginalFileName(),
                document.getModule() != null ? document.getModule().getNom() : "N/A",
                document.getModule() != null ? document.getModule().getId() : null,
                document.getUploadedBy() != null ? (document.getUploadedBy().getNom() + " " + document.getUploadedBy().getPrenom()) : "N/A",
                document.getUploadedAt(),
                downloadUrl
        );
    }

    @Transactional
    public DocumentResponse storeDocument(MultipartFile file, Long moduleId, Long uploaderUserId) {
        if (file.isEmpty()) {
            throw new BadRequestException("Failed to store empty file.");
        }
        if (!"application/pdf".equalsIgnoreCase(file.getContentType())) {
            throw new BadRequestException("Invalid file type. Only PDF files are allowed.");
        }

        String originalFileName = StringUtils.cleanPath(file.getOriginalFilename());
        // Generate a unique file name to prevent conflicts
        String uniqueFileName = UUID.randomUUID().toString() + "_" + originalFileName;

        try {
            if (uniqueFileName.contains("..")) {
                throw new FileStorageException("Filename contains invalid path sequence " + uniqueFileName);
            }

            Path targetLocation = this.fileStorageLocation.resolve(uniqueFileName);
            Files.copy(file.getInputStream(), targetLocation, StandardCopyOption.REPLACE_EXISTING);
            logger.info("File stored successfully: {} at path: {}", uniqueFileName, targetLocation.toString());

            User uploader = userService.findUserById(uploaderUserId); // findUserById should throw ResourceNotFound
            Module module = moduleService.findModuleEntityById(moduleId); // findModuleEntityById should throw ResourceNotFound

            Document document = new Document();
            document.setFileName(uniqueFileName);
            document.setOriginalFileName(originalFileName);
            document.setFilePath(targetLocation.toString()); // Store absolute or relative path as needed
            document.setContentType(file.getContentType());
            document.setSize(file.getSize());
            document.setModule(module);
            document.setUploadedBy(uploader);
            // uploadedAt is set by @CreationTimestamp

            Document savedDocument = documentRepository.save(document);
            logger.info("Document metadata saved for: {}", savedDocument.getOriginalFileName());

            // Optional: Trigger asynchronous RAG indexing
            // ragPipelineService.indexDocument(savedDocument.getId(), targetLocation.toString());

            return convertToDocumentResponse(savedDocument);

        } catch (IOException ex) {
            logger.error("Could not store file {}. Please try again!", originalFileName, ex);
            throw new FileStorageException("Could not store file " + originalFileName + ". Please try again!", ex);
        }
    }

    @Transactional(readOnly = true)
    public List<DocumentResponse> getDocumentsByUploader(Long uploaderUserId) {
        User uploader = userService.findUserById(uploaderUserId);
        logger.info("Fetching documents for uploader ID: {}", uploaderUserId);
        return documentRepository.findByUploadedBy(uploader).stream()
                .map(this::convertToDocumentResponse)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public List<DocumentResponse> getAllDocuments() {
        logger.info("Fetching all documents (admin view)");
        return documentRepository.findAll().stream()
                .map(this::convertToDocumentResponse)
                .collect(Collectors.toList());
    }


    @Transactional(readOnly = true)
    public Document findDocumentEntityById(Long documentId) {
        return documentRepository.findById(documentId)
                .orElseThrow(() -> new ResourceNotFoundException("Document", "id", documentId));
    }

    @Transactional(readOnly = true)
    public DocumentResponse getDocumentResponseById(Long documentId) {
        Document document = findDocumentEntityById(documentId);
        return convertToDocumentResponse(document);
    }


    @Transactional
    public DocumentResponse updateDocumentModule(Long documentId, Long newModuleId, Long currentUserId) {
        logger.info("Updating module for document ID: {} to module ID: {}", documentId, newModuleId);
        Document document = findDocumentEntityById(documentId);

        // Authorization check: Only the uploader can modify their document
        if (!document.getUploadedBy().getId().equals(currentUserId)) {
            logger.warn("User {} attempted to modify document {} owned by user {}",
                    currentUserId, documentId, document.getUploadedBy().getId());
            throw new AppException(HttpStatus.FORBIDDEN, "You do not have permission to modify this document.");
        }

        Module newModule = moduleService.findModuleEntityById(newModuleId);
        document.setModule(newModule);
        // updatedAt is updated by @UpdateTimestamp
        Document updatedDocument = documentRepository.save(document);
        logger.info("Document {} module updated successfully by user {}", documentId, currentUserId);
        return convertToDocumentResponse(updatedDocument);
    }

    @Transactional
    public void deleteDocument(Long documentId, Long currentUserId) {
        logger.info("Attempting to delete document ID: {} by user ID: {}", documentId, currentUserId);
        Document document = findDocumentEntityById(documentId);

        // Authorization check: Only the uploader (or an admin - add role check if needed) can delete
        User currentUser = userService.findUserById(currentUserId); // Get current user for role check if needed
        boolean isAdmin = currentUser.getRole().name().equals("ROLE_ADMINISTRATOR"); // Example

        if (!document.getUploadedBy().getId().equals(currentUserId) && !isAdmin) {
            logger.warn("User {} attempted to delete document {} owned by user {}",
                    currentUserId, documentId, document.getUploadedBy().getId());
            throw new AppException(HttpStatus.FORBIDDEN, "You do not have permission to delete this document.");
        }

        try {
            Path filePath = Paths.get(document.getFilePath());
            Files.deleteIfExists(filePath);
            logger.info("Physical file deleted: {}", document.getFilePath());

            documentRepository.delete(document);
            logger.info("Document metadata deleted for ID: {}", documentId);

            // Optional: Trigger asynchronous RAG de-indexing
            // ragPipelineService.deindexDocument(documentId);

        } catch (IOException ex) {
            logger.error("Error deleting physical file {}: {}", document.getFilePath(), ex.getMessage(), ex);
            // Decide if this should prevent metadata deletion or just be logged
            // For now, we'll throw, as inability to delete file is significant
            throw new FileStorageException("Could not delete file " + document.getOriginalFileName() + ". Error: " + ex.getMessage(), ex);
        }
    }

    public Resource loadDocumentAsResource(Long documentId) {
        logger.info("Loading document ID: {} as resource", documentId);
        Document document = findDocumentEntityById(documentId);
        try {
            Path filePath = Paths.get(document.getFilePath()); // Assuming filePath is absolute or resolvable
            // If filePath is relative to fileStorageLocation:
            // Path filePath = this.fileStorageLocation.resolve(document.getFileName()).normalize();

            Resource resource = new UrlResource(filePath.toUri());
            if (resource.exists() || resource.isReadable()) {
                return resource;
            } else {
                logger.error("Could not read file: {}", document.getOriginalFileName());
                throw new FileStorageException("Could not read file: " + document.getOriginalFileName());
            }
        } catch (MalformedURLException ex) {
            logger.error("Malformed URL for file {}: {}", document.getOriginalFileName(), ex.getMessage(), ex);
            throw new FileStorageException("Could not read file: " + document.getOriginalFileName(), ex);
        }
    }

    public String getOriginalDocumentName(Long documentId) {
        Document document = findDocumentEntityById(documentId);
        return document.getOriginalFileName();
    }
}