package com.fsbmchatbot.fsbmchatbotbackend.security; // Ensure your package is correct

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fsbmchatbot.fsbmchatbotbackend.model.User; // Import your User entity
import lombok.Getter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.io.Serial;
import java.util.Collection;
import java.util.Collections;
import java.util.Objects;

public class UserDetailsImpl implements UserDetails {
    @Serial
    private static final long serialVersionUID = 1L;


    @Getter
    private final Long id;
    private final String email; // This will be the "username" for Spring Security
    @JsonIgnore
    private  String password;

    @Getter
    private final String nom;

    @Getter
    private final String prenom;
    private final boolean isActive;  // Corrected from is_active to isActive to match Java conventions
    private final boolean isVerified; // Corrected from is_verified to isVerified

    private final Collection<? extends GrantedAuthority> authorities;

    public UserDetailsImpl(Long id, String email, String password, String nom, String prenom,
                           boolean isActive, boolean isVerified,
                           Collection<? extends GrantedAuthority> authorities) {
        this.id = id;
        this.email = email;
        this.password = password;
        this.nom = nom;
        this.prenom = prenom;
        this.isActive = isActive;
        this.isVerified = isVerified;
        this.authorities = authorities;
    }

    public static UserDetailsImpl build(User user) { // 'user' is the instance of your User entity
        // Ensure user.getRole() is not null before calling .name() if role can be null
        if (user.getRole() == null) {
            throw new IllegalArgumentException("User role cannot be null for building UserDetailsImpl");
        }
        GrantedAuthority authority = new SimpleGrantedAuthority(user.getRole().name());
        return new UserDetailsImpl(
                user.getId(),
                user.getEmail(),
                user.getPassword(), 
                user.getNom(),
                user.getPrenom(),
                user.isActive(),
                user.isVerified(),
                Collections.singletonList(authority));
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return authorities;
    }

    @Override
    public String getPassword() {
        return password;
    }

    @Override
    public String getUsername() {
        return email; // Spring Security uses getUsername()
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        // An account is "non-locked" if it's active.
        // If !isActive, it's effectively locked from the application's perspective.
        return this.isActive;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        // A user is fully "enabled" for Spring Security if they are both active AND verified.
        return this.isActive && this.isVerified;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        UserDetailsImpl that = (UserDetailsImpl) o;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

}