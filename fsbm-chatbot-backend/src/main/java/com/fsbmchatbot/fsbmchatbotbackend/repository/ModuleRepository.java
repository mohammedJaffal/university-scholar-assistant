package com.fsbmchatbot.fsbmchatbotbackend.repository;

import com.fsbmchatbot.fsbmchatbotbackend.model.Module;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ModuleRepository extends JpaRepository<Module, Long> {
    boolean existsByNom(@NotBlank(message = "Module name cannot be blank") @Size(max = 100) String nom);
    //Optional<Module> findByNom(String nom);
   // Boolean existsByNom(String nom);
}