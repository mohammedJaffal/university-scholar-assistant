package com.fsbmchatbot.fsbmchatbotbackend.repository;

import com.fsbmchatbot.fsbmchatbotbackend.model.Filiere;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface FiliereRepository extends JpaRepository<Filiere, Long> {
    Optional<Filiere> findByNom(String nom);

    boolean existsByNom(@NotBlank(message = "Filiere name cannot be blank") @Size(max = 50) String nom);
}