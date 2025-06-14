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
@Table(name = "filieres")
@Getter
@Setter
@NoArgsConstructor
public class Filiere {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // Optional: Setter for nom
    // Getter for nom
    @Setter
    @Getter
    @NotBlank
    @Size(max = 50)
    @Column(unique = true, nullable = false)
    private String nom; // e.g., "SMI", "SMA", "BCG"

    @Size(max = 255)
    private String description; // Optional

    @OneToMany(mappedBy = "filiere")
    private Set<User> users = new HashSet<>();

    public Filiere(String nom) {
        this.nom = nom;
    }

    public Filiere(String nom, String description) {
        this.nom = nom;
        this.description = description;
    }

}