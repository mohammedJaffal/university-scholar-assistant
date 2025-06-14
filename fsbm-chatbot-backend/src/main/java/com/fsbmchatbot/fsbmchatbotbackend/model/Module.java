package com.fsbmchatbot.fsbmchatbotbackend.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.HashSet;
import java.util.Set;

@Entity
@Table(name = "modules")
@Getter
@Setter
@NoArgsConstructor
public class Module {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank
    @Size(max = 100)
    @Column(unique = true, nullable = false)
    private String nom;

    @Size(max = 255)
    private String description;
    @Size(max = 255)
    private String code_module;
    @OneToMany(mappedBy = "module", cascade = CascadeType.PERSIST, orphanRemoval = false)
    private Set<Document> documents = new HashSet<>();

    public Module(String nom) {
        this.nom = nom;
    }

    public Module(String nom, String description) {
        this.nom = nom;
        this.description = description;
    }
}