package com.fsbmchatbot.fsbmchatbotbackend.repository;

import com.fsbmchatbot.fsbmchatbotbackend.model.Document;
import com.fsbmchatbot.fsbmchatbotbackend.model.Module; // If you add findByModule
import com.fsbmchatbot.fsbmchatbotbackend.model.User;   // This is the important import for findByUploadedBy
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional; // If you were to define a method returning a single optional document

@Repository
public interface DocumentRepository extends JpaRepository<Document, Long> {

    /**
     * Finds all documents uploaded by a specific user.
     * Spring Data JPA will automatically generate the query for this method
     * based on the "uploadedBy" field in the Document entity, which is of type User.
     *
     * @param user The User entity representing the uploader.
     * @return A list of Document entities uploaded by the specified user.
     */
    List<Document> findByUploadedBy(User user);

    // --- Other example derived queries you might have or add ---

    /**
     * Finds all documents associated with a specific module.
     *
     * @param module The Module entity.
     * @return A list of Document entities associated with the specified module.
     */
    List<Document> findByModule(Module module);

    /**
     * Finds all documents uploaded by a specific user and associated with a specific module.
     *
     * @param user The User entity representing the uploader.
     * @param module The Module entity.
     * @return A list of Document entities matching both criteria.
     */
    List<Document> findByUploadedByAndModule(User user, Module module);

    /**
     * Finds a document by its original file name (if you need such a query and originalFileName is unique).
     * If not unique, this should return List<Document>.
     *
     * @param originalFileName The original name of the file when it was uploaded.
     * @return An Optional containing the Document if found, or an empty Optional otherwise.
     */
    Optional<Document> findByOriginalFileName(String originalFileName);

    /**
     * Finds documents by the stored unique file name.
     * Since fileName is generated to be unique, this should return an Optional<Document>.
     *
     * @param fileName The unique name under which the file is stored on the server.
     * @return An Optional containing the Document if found.
     */
    Optional<Document> findByFileName(String fileName);

}