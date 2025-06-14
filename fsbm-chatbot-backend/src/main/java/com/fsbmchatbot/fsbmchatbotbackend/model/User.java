package com.fsbmchatbot.fsbmchatbotbackend.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

@Entity
@Table(name = "users", uniqueConstraints = {
        @UniqueConstraint(columnNames = "email", name = "uk_user_email"),
        @UniqueConstraint(columnNames = "emailUniversitaire", name = "uk_user_email_universitaire")
})
@Getter
@Setter
@NoArgsConstructor
@ToString(exclude = {"password", "uploadedDocuments", "verificationCode"})
@AllArgsConstructor
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @NotBlank(message = "Nom is mandatory")
    @Size(min = 2, max = 50)
    @Column(nullable = false, length = 50)
    private String nom;
    @NotBlank(message = "Prénom is mandatory")
    @Size(min = 2, max = 50)
    @Column(nullable = false, length = 50)
    private String prenom;
    @NotBlank(message = "Email is mandatory")
    @Email(message = "Email should be valid")
    @Size(max = 100)
    @Column(nullable = false, unique = true, length = 100)
    private String email;
    @Email(message = "University email should be valid")
    @Size(max = 100)
    @Column(unique = true, length = 100)
    private String emailUniversitaire;
    @NotBlank(message = "Password is mandatory")
    @Size(max = 120)
    @Column(nullable = false, length = 120)
    private String password;
    @Enumerated(EnumType.STRING)
    @Column(length = 20, nullable = false)
    private Role role;
    @Column(name = "filiere_nom")
    private String filiere;
    @Column(nullable = false)
    private boolean isActive = false;
    @Column(nullable = false)
    private boolean isVerified = false;
    @Column(length = 36)
    private String verificationCode;
    private LocalDateTime verificationCodeExpiry;
    @OneToMany(mappedBy = "uploadedBy", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    private Set<Document> uploadedDocuments = new HashSet<>();
    @CreationTimestamp
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;
    @UpdateTimestamp
    @Column(nullable = false)
    private LocalDateTime updatedAt;

    // --- Constructors ---

    public User(String nom, String prenom, String email, String password, Role role) {
        this.nom = nom;
        this.prenom = prenom;
        this.email = email;
        this.password = password;
        this.role = role;
        this.isActive = false;
        this.isVerified = false;
    }

    public User(String nom, String prenom, String email, String emailUniversitaire, String password, Role role, String filiere) {
        this.nom = nom;
        this.prenom = prenom;
        this.email = email;
        this.emailUniversitaire = emailUniversitaire;
        this.password = password;
        this.role = role;
        this.filiere = filiere;
        this.isActive = false;
        this.isVerified = false;
    }

    // --- Builder-enabled constructor ---
    @Builder
    public User(String nom, String prenom, String email, String emailUniversitaire, String password,
                Role role, String filiere, boolean isActive, boolean isVerified,
                String verificationCode, LocalDateTime verificationCodeExpiry) {
        this.nom = nom;
        this.prenom = prenom;
        this.email = email;
        this.emailUniversitaire = emailUniversitaire;
        this.password = password;
        this.role = role;
        this.filiere = filiere;
        this.isActive = isActive;
        this.isVerified = isVerified;
        this.verificationCode = verificationCode;
        this.verificationCodeExpiry = verificationCodeExpiry;
    }

    // --- Relationship helpers ---

    public void addDocument(Document document) {
        this.uploadedDocuments.add(document);
        document.setUploadedBy(this);
    }

    public void removeDocument(Document document) {
        this.uploadedDocuments.remove(document);
        document.setUploadedBy(null);
    }

    // --- Verification helpers ---

    public void setVerificationCode(String code) {
        this.verificationCode = code;
    }

    public LocalDateTime getVerificationCodeExpiry() {
        return this.verificationCodeExpiry;
    }

    public void setVerificationCodeExpiry(LocalDateTime expiry) {
        this.verificationCodeExpiry = expiry;
    }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        User user = (User) o;
        return id != null && Objects.equals(id, user.id);
    }

    @Override
    public int hashCode() {
        return id != null ? id.hashCode() : System.identityHashCode(this);
    }
}
