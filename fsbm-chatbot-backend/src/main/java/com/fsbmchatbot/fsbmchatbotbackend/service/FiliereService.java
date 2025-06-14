package com.fsbmchatbot.fsbmchatbotbackend.service;

import com.fsbmchatbot.fsbmchatbotbackend.dto.FiliereDTO;
import com.fsbmchatbot.fsbmchatbotbackend.exception.BadRequestException;
import com.fsbmchatbot.fsbmchatbotbackend.exception.ResourceNotFoundException;
import com.fsbmchatbot.fsbmchatbotbackend.model.Filiere;
import com.fsbmchatbot.fsbmchatbotbackend.repository.FiliereRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class FiliereService {

    private static final Logger logger = LoggerFactory.getLogger(FiliereService.class);

    @Autowired
    private FiliereRepository filiereRepository;

    private FiliereDTO convertToFiliereDTO(Filiere filiere) {
        return new FiliereDTO(
                filiere.getId(),
                filiere.getNom(),
                filiere.getDescription()
        );
    }

    @Transactional
    public FiliereDTO createFiliere(FiliereDTO filiereDTO) {
        if (filiereRepository.existsByNom(filiereDTO.getNom())) {
            throw new BadRequestException("Filiere with name '" + filiereDTO.getNom() + "' already exists.");
        }
        Filiere filiere = new Filiere();
        filiere.setNom(filiereDTO.getNom());
        filiere.setDescription(filiereDTO.getDescription());
        Filiere savedFiliere = filiereRepository.save(filiere);
        logger.info("Created new filiere: {}", savedFiliere.getNom());
        return convertToFiliereDTO(savedFiliere);
    }

    @Transactional(readOnly = true)
    public List<FiliereDTO> getAllFilieres() {
        logger.info("Fetching all filieres");
        return filiereRepository.findAll().stream()
                .map(this::convertToFiliereDTO)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public FiliereDTO getFiliereById(Long filiereId) {
        logger.info("Fetching filiere with ID: {}", filiereId);
        Filiere filiere = filiereRepository.findById(filiereId)
                .orElseThrow(() -> new ResourceNotFoundException("Filiere", "id", filiereId));
        return convertToFiliereDTO(filiere);
    }

    public Filiere findFiliereByNom(String nom) { // Used by AuthService
        return filiereRepository.findByNom(nom)
                .orElseThrow(() -> new ResourceNotFoundException("Filiere", "nom", nom));
    }

    @Transactional
    public FiliereDTO updateFiliere(Long filiereId, FiliereDTO filiereDTO) {
        logger.info("Updating filiere with ID: {}", filiereId);
        Filiere filiere = filiereRepository.findById(filiereId)
                .orElseThrow(() -> new ResourceNotFoundException("Filiere", "id", filiereId));

        if (!filiere.getNom().equals(filiereDTO.getNom()) && filiereRepository.existsByNom(filiereDTO.getNom())) {
            throw new BadRequestException("Another filiere with name '" + filiereDTO.getNom() + "' already exists.");
        }

        filiere.setNom(filiereDTO.getNom());
        filiere.setDescription(filiereDTO.getDescription());
        Filiere updatedFiliere = filiereRepository.save(filiere);
        logger.info("Filiere {} updated successfully", updatedFiliere.getNom());
        return convertToFiliereDTO(updatedFiliere);
    }

    @Transactional
    public void deleteFiliere(Long filiereId) {
        logger.info("Deleting filiere with ID: {}", filiereId);
        Filiere filiere = filiereRepository.findById(filiereId)
                .orElseThrow(() -> new ResourceNotFoundException("Filiere", "id", filiereId));

        if (!filiere.getUsers().isEmpty()) {
            throw new BadRequestException("Cannot delete filiere '" + filiere.getNom() + "' as it has associated users.");
        }

        filiereRepository.delete(filiere);
        logger.info("Filiere with ID {} deleted successfully", filiereId);
    }
}