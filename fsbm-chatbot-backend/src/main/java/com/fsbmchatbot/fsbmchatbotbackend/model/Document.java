package com.fsbmchatbot.fsbmchatbotbackend.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "documents")
@Getter
@Setter
@NoArgsConstructor
public class Document {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank
    private String fileName;

    @NotBlank
    private String originalFileName;

    @NotBlank
    private String filePath;

    @NotBlank
    private String contentType;

    private long size;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "module_id", nullable = false)
    private Module module;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "uploaded_by_user_id", nullable = false)
    private User uploadedBy;

    @CreationTimestamp
    private LocalDateTime uploadedAt;

    public Document(String fileName, String originalFileName, String filePath, String contentType, long size, Module module, User uploadedBy) {
        this.fileName = fileName;
        this.originalFileName = originalFileName;
        this.filePath = filePath;
        this.contentType = contentType;
        this.size = size;
        this.module = module;
        this.uploadedBy = uploadedBy;
    }

    public void setUploadedBy(User aThis) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

}